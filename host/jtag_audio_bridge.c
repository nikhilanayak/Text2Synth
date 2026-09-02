/* Raw Intel JTAG Atlantic reader. Build with the Quartus-provided library. */
#include <stdio.h>
#include <stdlib.h>
#include "jtag_atlantic.h"

int main(int argc, char **argv) {
    const char *cable = argc > 1 ? argv[1] : NULL;
    int device = argc > 2 ? atoi(argv[2]) : -1;
    int instance = argc > 3 ? atoi(argv[3]) : -1;
    JTAGATLANTIC *link = jtagatlantic_open(cable, device, instance, "ctag-audio");
    unsigned char buffer[4096];
    if (!link) {
        fprintf(stderr, "unable to open Intel JTAG UART\n");
        return 2;
    }
    for (;;) {
        int count = jtagatlantic_read(link, (char *)buffer, sizeof buffer);
        if (count < 0) break;
        if (count && fwrite(buffer, 1, (size_t)count, stdout) != (size_t)count) break;
        fflush(stdout);
    }
    jtagatlantic_close(link);
    return 0;
}
