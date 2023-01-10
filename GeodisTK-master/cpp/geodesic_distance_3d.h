#include <iostream>
#include <vector>
using namespace std;

struct Point3D
{
    float distance;
    int d;
    int w;
    int h;
};

void geodesic3d_fast_marching(const float * img, const unsigned char * seeds, float * distance,
                         int depth, int height, int width, int channel, std::vector<float> spacing);

void geodesic3d_raster_scan(const float * img, const unsigned char * seeds, float * distance,
                            int depth, int height, int width, int channel, 
                            std::vector<float> spacing, float lambda, int iteration);
