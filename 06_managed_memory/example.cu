#include <stdio.h>

class Managed {
public: 
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    } 

    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};


class umString : public Managed {
public:
    int length;
    char *data;
    umString (const umString &s) {
        length = s.length;
        cudaMallocManaged(&data, length);
        memcpy(data, s.data, length);
    }

    umString (const char *s) {
        length = strlen(s);
        cudaMallocManaged(&data, length);
        memcpy(data, s, length);
    }
};


class dataElem : public Managed {
public:
    int key;
    umString name;

    dataElem() : key(0), name("hello") {}
};


__global__ void kernel(dataElem *data) {
    data[threadIdx.x].key = threadIdx.x;
    data[threadIdx.x].name = umString("world");
}

int main() {
    dataElem *data = new dataElem[100];
    kernel<<<1, 100>>>(data);
    cudaDeviceSynchronize();
    for (int i = 0; i < 100; i++) {
        printf("%d: %s\n", data[i].key, data[i].name.data);
    }
    delete[] data;
    return 0;
}