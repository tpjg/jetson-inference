/*
 * (c) Sensys Gatso Managed Services BV
 *
 * Code adapted from http://github.com/dusty-nv/jetson-inference
 *
 * The "detectnet-console" application is modified so instead it will read a
 * UYUV image from shared (CPU) memory, copy it to CUDA space, convert UYUV
 * to RGBA, then run the detectNet. Minor modification is also made to do
 * conversion from uchar4 RGBA to float4 while doing resizing to the input size
 * of the neural network (alternative to cudaPreImageNetMean).
 *
 * The application reads the shared memory after receiving a trigger from Redis,
 * this trigger is fired whenever a new frame is available.
 *
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"
#include "cudaYUV.h"

#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

#define WITH_REDIS
#ifdef WITH_REDIS
#include "hiredis/hiredis.h"
#include "hiredis/async.h"
#include "hiredis/adapters/libevent.h"
#endif


uint64_t current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

// Map shared memory segment
void* mapFile(const char *fname, off_t *fsize) {
  if ((fname == NULL) || (fsize == NULL)){
    return NULL;
  }
  // Open segment and map into memory
  int fd = shm_open(fname, O_RDWR, 0666);
  if (fd<0) {
    printf("Cannot open %s memory segment - %s\n", fname, strerror(errno));
    return NULL;
  }
  *fsize = lseek(fd, 0, SEEK_END);
  //printf("Opened %s : %ld bytes\n", fname, *fsize);
  void *data = mmap(NULL, *fsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (data == MAP_FAILED) {
    printf("Cannot map file into memory - %s\n", strerror(errno));
    return NULL;
  }
}


// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-console\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");


	// retrieve filename argument
	if( argc < 2 )
	{
		printf("detectnet-console:   input image filename required\n");
		return 0;
	}

	const char* imgFilename = argv[1];


	// create detectNet
	//detectNet* net = detectNet::Create( detectNet::PEDNET_MULTI ); // uncomment to enable one of these
  //detectNet* net = detectNet::Create( detectNet::PEDNET );
  //detectNet* net = detectNet::Create( detectNet::FACENET );
  detectNet* net = detectNet::Create("deploy.prototxt", "deploy.caffemodel", "mean.binaryproto" );

	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}

	net->EnableProfiler();

	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();

	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;

	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}

	// map image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 2048;
	int    imgHeight = 1088;
  off_t  fSize = 0;

  void* frame = NULL;
  frame = mapFile(imgFilename, &fSize);
  if (!frame){
		printf("failed to load image '%s'\n", imgFilename);
		return 1;
	}
  if ((imgWidth*imgHeight*2)>fSize) {
    printf("mapped file not big enough for %dx%d image\n", imgWidth, imgHeight);
    return 1;
  }

  // Allocate buffers
  uchar2* uyvyCUDA = NULL;
  if ( CUDA_FAILED(cudaMalloc((void**)&uyvyCUDA, fSize)) ){
    printf("error allocating memory on CUDA device for UYVY\n");
    return 1;
  }
  printf("uyvyCUDA @ %p\n", uyvyCUDA);
  uchar4* rgbaCUDA = NULL;
  if ( CUDA_FAILED(cudaMalloc((void**)&rgbaCUDA, imgWidth*imgHeight*4)) ){
    printf("error allocating memory on CUDA device for RGBA\n");
    return 1;
  }
  printf("rgbaCUDA @ %p\n", rgbaCUDA);

// Wait for Redis to publish
#ifdef WITH_REDIS
  redisContext *c = redisConnect("127.0.0.1", 6379);
  if (c == NULL || c->err) {
      if (c) {
          printf("Error: %s\n", c->errstr);
          // handle error
      } else {
          printf("Can't allocate redis context\n");
      }
  }
  redisReply *r = NULL;

  redisContext *c2 = redisConnect("127.0.0.1", 6379);
  if (c2 == NULL || c->err) {
    if (c2) {
      printf("Error: %s\n", c2->errstr);
      // handle error
    } else {
      printf("Can't allocate redis context\n");
    }
  }

for (int loop=0; loop<1000; loop++){

  r = (redisReply*)redisCommand(c,"SUBSCRIBE FrameCreated");
  freeReplyObject(r);
  if (redisGetReply(c,(void**)&r) == REDIS_OK) {
    if (r->type == REDIS_REPLY_ARRAY) {
    // Check for "FrameCreated" message
    if ((r->elements == 3) && (strncmp(r->element[0]->str,"message",7)==0) && (strncmp(r->element[1]->str,"FrameCreated",12)==0)) {
        char *hashname = r->element[2]->str;

        redisReply *reply = (redisReply*)redisCommand(c2,"HGET %s MemorySegment", hashname); // Get the memory segment from the hash
        imgFilename = reply->str;
        printf("FrameCreated(%s) -> MemorySegment: %s\n", hashname,  imgFilename);

        frame = mapFile(imgFilename, &fSize);
        if (!frame){
          printf("failed to load image '%s'\n", imgFilename);
          return 1;
        }

        freeReplyObject(reply);
    }

    freeReplyObject(r);
    }
    r = (redisReply*)redisCommand(c,"UNSUBSCRIBE FrameCreated");
    freeReplyObject(r);
} else {
  printf("Error when trying to SUBSCRIBE to FrameCreated\n");
}
#endif

  // Copy to CUDA memory and UYVY to RGBA
  if ( CUDA_FAILED(cudaMemcpy((void*)uyvyCUDA, frame, fSize, cudaMemcpyHostToDevice)) ){
    printf("error copying data to CUDA device\n");
    return 1;
  }
  if ( CUDA_FAILED(cudaUYVYToRGBA(uyvyCUDA, imgWidth*2, rgbaCUDA, imgWidth*4, imgWidth, imgHeight)) ) {
    printf("error converting UYVY to RGBA\n");
    return 1;
  }

	// classify image
	int numBoundingBoxes = maxBoxes;

	printf("detectnet-console:  beginning processing network (%zu)\n", current_timestamp());

	const bool result = net->DetectChar((char*)rgbaCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);

	printf("detectnet-console:  finished processing network  (%zu)\n", current_timestamp());

	if( !result )
		printf("detectnet-console:  failed to classify '%s'\n", imgFilename);
	else
	{
		printf("%i bounding boxes detected\n", numBoundingBoxes);

		int lastClass = 0;
		int lastStart = 0;

		for( int n=0; n < numBoundingBoxes; n++ )
		{
			const int nc = confCPU[n*2+1];
			float* bb = bbCPU + (n * 4);

			printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);

			if( nc != lastClass || n == (numBoundingBoxes - 1) )
			{
        // Draw directly onto uyuv image (shared in memory)
        uint8_t* img = (uint8_t*)frame;
        for ( int y=(bb[1]); y<bb[3]; y++){
        uint8_t* line1 = &img[int(bb[1])*imgWidth*2];
        uint8_t* line2 = &img[int(bb[3])*imgWidth*2];
        uint8_t* line = &img[y*imgWidth*2];
        for ( int x=int(bb[0]); x <int(bb[2]); x+=2) {
          line[x] = line[x]/2;
          line1[x] = 0;
          line2[x] = 0;
        }
      }
#if 0
				if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
					printf("detectnet-console:  failed to draw boxes\n");
#endif

				lastClass = nc;
				lastStart = n;
			}
		}

		//CUDA(cudaThreadSynchronize());
	}

#ifdef WITH_REDIS
} // end of loop
#endif

	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
