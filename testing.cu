#include "functions.cuh"
#include "fileops.h"
#include "diffeq.cuh"

int main() {
    param p = param();
    p.base_voltage = 160;
    printf("%f * (223399 - 208310) / (%f * %f) = %f\n", p.c23, p.c24, p.base_voltage + p.voltage_offset, p.c23 * (223399 - 208310) / ((p.base_voltage + p.voltage_offset) * p.c24));
    return 0;
}