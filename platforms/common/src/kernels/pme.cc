KERNEL void findAtomGridIndex(GLOBAL const real4* RESTRICT posq, GLOBAL int2* RESTRICT pmeAtomGridIndex,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ
#ifdef USE_ESP_OUTPUT_TILE_SPREAD
        , GLOBAL real4* RESTRICT espOutputTileAtomGridPosition
#endif
    ) {
    // Compute the index of the grid point each atom is associated with.

    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
#ifdef USE_ESP_OUTPUT_TILE_SPREAD
        int binX = gridIndex.x/ESP_OUTPUT_TILE_BIN_SIZE;
        int binY = gridIndex.y/ESP_OUTPUT_TILE_BIN_SIZE;
        int binZ = gridIndex.z/ESP_OUTPUT_TILE_BIN_SIZE;
        pmeAtomGridIndex[atom] = make_int2(atom, (binX*ESP_OUTPUT_TILE_NUM_BINS_Y+binY)*ESP_OUTPUT_TILE_NUM_BINS_Z+binZ);
        espOutputTileAtomGridPosition[atom] = make_real4(t.x, t.y, t.z, pos.w);
#else
        pmeAtomGridIndex[atom] = make_int2(atom, gridIndex.x*GRID_SIZE_Y*GRID_SIZE_Z+gridIndex.y*GRID_SIZE_Z+gridIndex.z);
#endif
    }
}

#ifdef USE_ESP
#define SPREAD_ORDER ESP_ORDER

DEVICE inline real3 espHorner3(GLOBAL const real* RESTRICT coeffs, int index, int stride, int order, real3 x) {
    real c0 = coeffs[index];
    real3 value = make_real3(c0, c0, c0);
    for (int i = 1; i < order; i++) {
        real c = coeffs[i*stride+index];
        value.x = value.x*x.x + c;
        value.y = value.y*x.y + c;
        value.z = value.z*x.z + c;
    }
    return value;
}

#ifdef USE_ESP_OUTPUT_TILE_SPREAD

KERNEL void initOutputTileBins(GLOBAL int* RESTRICT binSize) {
    for (int bin = GLOBAL_ID; bin < ESP_OUTPUT_TILE_NUM_BINS; bin += GLOBAL_SIZE)
        binSize[bin] = 0;
}

KERNEL void countOutputTileBinSizes(GLOBAL const int2* RESTRICT pmeAtomGridIndex, GLOBAL int* RESTRICT binSize,
        GLOBAL int* RESTRICT sortIndex) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        int bin = pmeAtomGridIndex[atom].y;
        sortIndex[atom] = ATOMIC_ADD(&binSize[bin], 1);
    }
}

KERNEL void scanOutputTileBinSizes(GLOBAL const int* RESTRICT binSize, GLOBAL int* RESTRICT binStart,
        GLOBAL int* RESTRICT scanBlockSums) {
    LOCAL int temp[ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE];
    int local = LOCAL_ID;
    int index = GROUP_ID*ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE+local;
    int value = (index < ESP_OUTPUT_TILE_NUM_BINS ? binSize[index] : 0);
    temp[local] = value;
    SYNC_THREADS

    for (int offset = 1; offset < ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE; offset <<= 1) {
        int add = (local >= offset ? temp[local-offset] : 0);
        SYNC_THREADS
        temp[local] += add;
        SYNC_THREADS
    }
    if (index < ESP_OUTPUT_TILE_NUM_BINS)
        binStart[index] = temp[local]-value;
    if (local == ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE-1)
        scanBlockSums[GROUP_ID] = temp[local];
}

KERNEL void scanOutputTileBlockSums(GLOBAL const int* RESTRICT scanBlockSums,
        GLOBAL int* RESTRICT scanBlockOffsets) {
    LOCAL int temp[ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE];
    int local = LOCAL_ID;
    int value = (local < ESP_OUTPUT_TILE_NUM_SCAN_BLOCKS ? scanBlockSums[local] : 0);
    temp[local] = value;
    SYNC_THREADS

    for (int offset = 1; offset < ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE; offset <<= 1) {
        int add = (local >= offset ? temp[local-offset] : 0);
        SYNC_THREADS
        temp[local] += add;
        SYNC_THREADS
    }
    if (local < ESP_OUTPUT_TILE_NUM_SCAN_BLOCKS)
        scanBlockOffsets[local] = temp[local]-value;
}

KERNEL void scatterOutputTileSortedAtoms(GLOBAL const int2* RESTRICT pmeAtomGridIndex,
        GLOBAL const int* RESTRICT binStart, GLOBAL const int* RESTRICT scanBlockOffsets,
        GLOBAL const int* RESTRICT sortIndex, GLOBAL int* RESTRICT sortedAtoms) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        int bin = pmeAtomGridIndex[atom].y;
        int block = bin/ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE;
        sortedAtoms[binStart[bin]+scanBlockOffsets[block]+sortIndex[atom]] = atom;
    }
}

DEVICE inline int espOutputTileWrapIndex(int index, int size) {
    return index < size ? index : index-size;
}

#endif

#else
#define SPREAD_ORDER PME_ORDER

#endif

KERNEL void gridSpreadCharge(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real* RESTRICT pmeGrid,
#endif
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
#ifdef USE_ESP
        , GLOBAL const real* RESTRICT espSpreadCoeffs
#endif
        ) {
    // Process the atoms in spatially sorted order.  This improves efficiency when writing
    // the grid values.  SPREAD_ORDER threads process one atom.

    real3 data[SPREAD_ORDER];
#ifndef USE_ESP
    const real scale = RECIP((real) (PME_ORDER-1));
#endif
    for (int i = GLOBAL_ID; i < NUM_ATOMS*SPREAD_ORDER; i += GLOBAL_SIZE) {
        int atom = pmeAtomGridIndex[i/SPREAD_ORDER].x;
        real4 pos = posq[atom];
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        if (charge == 0)
            continue;

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
#ifdef USE_ESP
        // ESP_ORDER is a compile-time define selected when this Context builds the kernel.
        // It can differ between Contexts/tolerances, but not within an initialized kernel.
        for (int j = 0; j < ESP_ORDER; j++)
            data[j] = espHorner3(espSpreadCoeffs, j, ESP_ORDER, ESP_POLY_ORDER, dr);
#else
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
#endif

        // Spread the charge from this atom onto each grid point.  SPREAD_ORDER threads access
        // consecutive addresses.

        int iz = i%SPREAD_ORDER;
        int zindex = gridIndex.z+iz;
        zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
        real dz = 0;
        for (int i = 0; i < SPREAD_ORDER; i++) {
            dz = i == iz ? data[i].z : dz;
        }
        dz *= charge;
        for (int ix = 0; ix < SPREAD_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dzdx = dz*data[ix].x;
            for (int iy = 0; iy < SPREAD_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = ybase*GRID_SIZE_Z;
                int index = xbase + ybase + zindex;
                real add = dzdx*data[iy].y;
                if (fabs(add) > 2.3e-10f) { // Smallest value representable in 64 bit fixed point
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add));
#if defined(__GFX12__) && defined(USE_HIP)
                    // Workaround for rare cases when few values of pmeGrid are very large and
                    // incorrect. The cause is unknown. Why this workaround or other irrelevant
                    // changes like printf help is also unknown.
                    asm volatile("s_wait_storecnt 0x0");
#endif
#else
                    ATOMIC_ADD(&pmeGrid[index], add);
#endif
                }
            }
        }
    }
}

#if defined(USE_ESP) && defined(USE_ESP_OUTPUT_TILE_SPREAD)

KERNEL void gridSpreadChargeOutputTile(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real* RESTRICT pmeGrid,
#endif
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
        GLOBAL const real4* RESTRICT espOutputTileAtomGridPosition, GLOBAL const int* RESTRICT binSize,
        GLOBAL const int* RESTRICT binStart, GLOBAL const int* RESTRICT scanBlockOffsets,
        GLOBAL const int* RESTRICT sortedAtoms,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon,
#else
        GLOBAL const real* RESTRICT charges,
#endif
        GLOBAL const real* RESTRICT espSpreadCoeffs) {
#ifdef USE_DOUBLE_PRECISION
    LOCAL real localGrid[ESP_OUTPUT_TILE_LOCAL_SIZE];
#else
    LOCAL real2 localGrid[ESP_OUTPUT_TILE_LOCAL_SIZE];
#endif
    LOCAL int3 localStart[ESP_OUTPUT_TILE_NP];
    LOCAL real chargeData[ESP_OUTPUT_TILE_NP];
    LOCAL real thetaX[ESP_OUTPUT_TILE_NP*ESP_ORDER];
    LOCAL real thetaY[ESP_OUTPUT_TILE_NP*ESP_ORDER];
    LOCAL real thetaZ[ESP_OUTPUT_TILE_NP*ESP_ORDER];

    if (GROUP_ID >= ESP_OUTPUT_TILE_NUM_BINS)
        return;
    int bin = GROUP_ID;
    int block = bin/ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE;
    int segmentStart = binStart[bin]+scanBlockOffsets[block];
    int segmentLimit = binStart[bin]+scanBlockOffsets[block]+binSize[bin];
    int segmentEnd = segmentLimit;

    int tmp = bin;
    int binZ = tmp%ESP_OUTPUT_TILE_NUM_BINS_Z;
    tmp /= ESP_OUTPUT_TILE_NUM_BINS_Z;
    int binY = tmp%ESP_OUTPUT_TILE_NUM_BINS_Y;
    int binX = tmp/ESP_OUTPUT_TILE_NUM_BINS_Y;
    int3 binOffset = make_int3(binX*ESP_OUTPUT_TILE_BIN_SIZE, binY*ESP_OUTPUT_TILE_BIN_SIZE, binZ*ESP_OUTPUT_TILE_BIN_SIZE);

    for (int i = LOCAL_ID; i < ESP_OUTPUT_TILE_LOCAL_SIZE; i += LOCAL_SIZE)
#ifdef USE_DOUBLE_PRECISION
        localGrid[i] = 0;
#else
        localGrid[i] = make_real2(0, 0);
#endif
    SYNC_THREADS

    for (int batchStart = segmentStart; batchStart < segmentEnd; batchStart += ESP_OUTPUT_TILE_NP) {
        int batchSize = segmentEnd-batchStart;
        batchSize = (batchSize < ESP_OUTPUT_TILE_NP ? batchSize : ESP_OUTPUT_TILE_NP);
        if (LOCAL_ID < batchSize) {
            int sortedIndex = batchStart+LOCAL_ID;
            int atom = sortedAtoms[sortedIndex];
            real4 t = espOutputTileAtomGridPosition[atom];
            real4 pos = make_real4(0, 0, 0, t.w);
#ifdef CHARGE_FROM_SIGEPS
            const float2 sigEps = sigmaEpsilon[atom];
            const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
            const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
            chargeData[LOCAL_ID] = charge;
            int3 gridIndex = make_int3((int) t.x, (int) t.y, (int) t.z);
            localStart[LOCAL_ID] = make_int3(gridIndex.x-binOffset.x, gridIndex.y-binOffset.y, gridIndex.z-binOffset.z);
            real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
#pragma unroll
            for (int j = 0; j < ESP_ORDER; j++) {
                real3 theta = espHorner3(espSpreadCoeffs, j, ESP_ORDER, ESP_POLY_ORDER, dr);
                thetaX[LOCAL_ID*ESP_ORDER+j] = theta.x;
                thetaY[LOCAL_ID*ESP_ORDER+j] = theta.y;
                thetaZ[LOCAL_ID*ESP_ORDER+j] = theta.z;
            }
        }
        SYNC_THREADS

        const int stencilSize = ESP_ORDER*ESP_ORDER*ESP_ORDER;
        for (int p = 0; p < batchSize; p++) {
            real charge = chargeData[p];
            int3 start = localStart[p];
            for (int index = LOCAL_ID; index < stencilSize; index += LOCAL_SIZE) {
                int tmp = index;
                int ix = tmp%ESP_ORDER;
                int localIndex = (start.x+ix)*ESP_OUTPUT_TILE_PADDED_SIZE*ESP_OUTPUT_TILE_PADDED_SIZE;
                real add = thetaX[p*ESP_ORDER+ix];
                tmp /= ESP_ORDER;
                int iy = tmp%ESP_ORDER;
                localIndex += ESP_OUTPUT_TILE_PADDED_SIZE*(start.y+iy);
                add *= thetaY[p*ESP_ORDER+iy];
                tmp /= ESP_ORDER;
                int iz = tmp;
                localIndex += start.z+iz;
                add *= charge*thetaZ[p*ESP_ORDER+iz];
#ifdef USE_DOUBLE_PRECISION
                localGrid[localIndex] += add;
#else
                localGrid[localIndex].x += add;
#endif
            }
            SYNC_THREADS
        }
    }

    for (int i = LOCAL_ID; i < ESP_OUTPUT_TILE_LOCAL_SIZE; i += LOCAL_SIZE) {
#ifdef USE_DOUBLE_PRECISION
        real add = localGrid[i];
#else
        real add = localGrid[i].x;
#endif
        int tmp = i;
        int localZ = tmp%ESP_OUTPUT_TILE_PADDED_SIZE;
        tmp /= ESP_OUTPUT_TILE_PADDED_SIZE;
        int localY = tmp%ESP_OUTPUT_TILE_PADDED_SIZE;
        int localX = tmp/ESP_OUTPUT_TILE_PADDED_SIZE;
        int x = espOutputTileWrapIndex(binOffset.x+localX, GRID_SIZE_X);
        int y = espOutputTileWrapIndex(binOffset.y+localY, GRID_SIZE_Y);
        int z = espOutputTileWrapIndex(binOffset.z+localZ, GRID_SIZE_Z);
        int gridIndex = (x*GRID_SIZE_Y+y)*GRID_SIZE_Z+z;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        ATOMIC_ADD(&pmeGrid[gridIndex], (mm_ulong) realToFixedPoint(add));
#else
        ATOMIC_ADD(&pmeGrid[gridIndex], add);
#endif
    }
}

#ifdef USE_ESP_OUTPUT_TILE_INTERP

KERNEL void gridInterpolateForceOutputTile(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ,
        GLOBAL const real4* RESTRICT espOutputTileAtomGridPosition, GLOBAL const int* RESTRICT binSize,
        GLOBAL const int* RESTRICT binStart, GLOBAL const int* RESTRICT scanBlockOffsets,
        GLOBAL const int* RESTRICT sortedAtoms,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon,
#else
        GLOBAL const real* RESTRICT charges,
#endif
        GLOBAL const real* RESTRICT espSpreadCoeffs, GLOBAL const real* RESTRICT espSpreadDerCoeffs) {
    LOCAL real localGrid[ESP_OUTPUT_TILE_LOCAL_SIZE];
    real3 data[ESP_ORDER];
    real3 ddata[ESP_ORDER];

    int bin = GROUP_ID;
    int block = bin/ESP_OUTPUT_TILE_SCAN_BLOCK_SIZE;
    int segmentStart = binStart[bin]+scanBlockOffsets[block];
    int segmentSize = binSize[bin];
    if (segmentSize == 0)
        return;
    int segmentEnd = segmentStart+segmentSize;

    int tmp = bin;
    int binZ = tmp%ESP_OUTPUT_TILE_NUM_BINS_Z;
    tmp /= ESP_OUTPUT_TILE_NUM_BINS_Z;
    int binY = tmp%ESP_OUTPUT_TILE_NUM_BINS_Y;
    int binX = tmp/ESP_OUTPUT_TILE_NUM_BINS_Y;
    int3 binOffset = make_int3(binX*ESP_OUTPUT_TILE_BIN_SIZE, binY*ESP_OUTPUT_TILE_BIN_SIZE, binZ*ESP_OUTPUT_TILE_BIN_SIZE);

    for (int i = LOCAL_ID; i < ESP_OUTPUT_TILE_LOCAL_SIZE; i += LOCAL_SIZE) {
        int tmp = i;
        int localZ = tmp%ESP_OUTPUT_TILE_PADDED_SIZE;
        tmp /= ESP_OUTPUT_TILE_PADDED_SIZE;
        int localY = tmp%ESP_OUTPUT_TILE_PADDED_SIZE;
        int localX = tmp/ESP_OUTPUT_TILE_PADDED_SIZE;
        int x = espOutputTileWrapIndex(binOffset.x+localX, GRID_SIZE_X);
        int y = espOutputTileWrapIndex(binOffset.y+localY, GRID_SIZE_Y);
        int z = espOutputTileWrapIndex(binOffset.z+localZ, GRID_SIZE_Z);
        localGrid[i] = pmeGrid[(x*GRID_SIZE_Y+y)*GRID_SIZE_Z+z];
    }
    SYNC_THREADS

    for (int sortedIndex = segmentStart+LOCAL_ID; sortedIndex < segmentEnd; sortedIndex += LOCAL_SIZE) {
        int atom = sortedAtoms[sortedIndex];
        real4 t = espOutputTileAtomGridPosition[atom];
        real4 pos = make_real4(0, 0, 0, t.w);
        real3 force = make_real3(0);
        int3 gridIndex = make_int3((int) t.x, (int) t.y, (int) t.z);
        int3 localStart = make_int3(gridIndex.x-binOffset.x, gridIndex.y-binOffset.y, gridIndex.z-binOffset.z);
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        real q = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        real q = CHARGE*EPSILON_FACTOR;
#endif
        if (q == 0)
            continue;

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        for (int j = 0; j < ESP_ORDER; j++)
            data[j] = espHorner3(espSpreadCoeffs, j, ESP_ORDER, ESP_POLY_ORDER, dr);
        for (int j = 0; j < ESP_ORDER; j++)
            ddata[j] = espHorner3(espSpreadDerCoeffs, j, ESP_ORDER, ESP_DER_POLY_ORDER, dr);

        for (int ix = 0; ix < ESP_ORDER; ix++) {
            int xbase = (localStart.x+ix)*ESP_OUTPUT_TILE_PADDED_SIZE*ESP_OUTPUT_TILE_PADDED_SIZE;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;
            for (int iy = 0; iy < ESP_ORDER; iy++) {
                int ybase = xbase + (localStart.y+iy)*ESP_OUTPUT_TILE_PADDED_SIZE;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;
                real zsum = 0;
                real dzsum = 0;
                for (int iz = 0; iz < ESP_ORDER; iz++) {
                    real gridvalue = localGrid[ybase+localStart.z+iz];
                    zsum += data[iz].z*gridvalue;
                    dzsum += ddata[iz].z*gridvalue;
                }
                force.x += ddx*dy*zsum;
                force.y += dx*ddy*zsum;
                force.z += dx*dy*dzsum;
            }
        }
        real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
        real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
        real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
#ifdef USE_PME_STREAM
        ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(forceX));
        ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceY));
        ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceZ));
#else
        forceBuffers[atom] += (mm_ulong) realToFixedPoint(forceX);
        forceBuffers[atom+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceY);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceZ);
#endif
    }
}

#endif

#endif

#ifdef USE_FIXED_POINT_CHARGE_SPREADING

KERNEL void finishSpreadCharge(
        GLOBAL const mm_long* RESTRICT grid1,
        GLOBAL real* RESTRICT grid2) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        grid2[index] = scale*grid1[index];
    }
}

#endif

KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
#ifdef USE_LJPME
    const real recipScaleFactor = -(2*M_PI/6)*SQRT(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#elif defined(USE_ESP)
    const real recipScaleFactor = ((real) 0.5/(real) M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#else
    const real recipScaleFactor = RECIP(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#endif

    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
        int ky = remainder/(GRID_SIZE_Z/2+1);
        int kz = remainder-ky*(GRID_SIZE_Z/2+1);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
#ifdef USE_ESP
        if (kx != 0 || ky != 0 || kz != 0) {
            real arg = SQRT(m2)*(real) ESP_ARG_SCALE;
            real splitVal = (arg > (real) 1 ? (real) 0 : (ESP_SPLIT_POLY));
            real eterm = recipScaleFactor*splitVal*bx*by*bz*RECIP(m2);
            pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
        }
        else
            pmeGrid[index] = make_real2(0);
#elif defined(USE_LJPME)
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        real eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
        pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
#else
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
        if (kx != 0 || ky != 0 || kz != 0) {
            pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
        } else {
            pmeGrid[index] = make_real2(0);
        }
#endif
    }
}

KERNEL void gridEvaluateEnergy(GLOBAL real2* RESTRICT pmeGrid, GLOBAL mixed* RESTRICT energyBuffer,
                      GLOBAL const real* RESTRICT pmeBsplineModuliX, GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ,
                      real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
 #ifdef USE_LJPME
    const real recipScaleFactor = -(2*M_PI/6)*SQRT(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#elif defined(USE_ESP)
    const real recipScaleFactor = ((real) 0.5/(real) M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#else
    const real recipScaleFactor = RECIP(M_PI)*recipBoxVecX.x*recipBoxVecY.y*recipBoxVecZ.z;
#endif

    mixed energy = 0;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z);
        int ky = remainder/(GRID_SIZE_Z);
        int kz = remainder-ky*(GRID_SIZE_Z);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real eterm = 0;
#ifdef USE_ESP
        if (kx != 0 || ky != 0 || kz != 0) {
            real arg = SQRT(m2)*(real) ESP_ARG_SCALE;
            real splitVal = (arg > (real) 1 ? (real) 0 : (ESP_SPLIT_POLY));
            eterm = recipScaleFactor*splitVal*bx*by*bz*RECIP(m2);
        }
#elif defined(USE_LJPME)
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
#else
        real denom = m2*bx*by*bz;
        eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
#endif
        if (kz >= (GRID_SIZE_Z/2+1)) {
            kx = ((kx == 0) ? kx : GRID_SIZE_X-kx);
            ky = ((ky == 0) ? ky : GRID_SIZE_Y-ky);
            kz = GRID_SIZE_Z-kz;
        }
        int indexInHalfComplexGrid = kz + ky*(GRID_SIZE_Z/2+1)+kx*(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        real2 grid = pmeGrid[indexInHalfComplexGrid];
#if !defined(USE_LJPME) && !defined(USE_ESP)
        if (kx != 0 || ky != 0 || kz != 0)
#endif
            energy += eterm*(grid.x*grid.x + grid.y*grid.y);
    }
#if defined(USE_PME_STREAM) && !defined(USE_LJPME)
    energyBuffer[GLOBAL_ID] = 0.5f*energy;
#else
    energyBuffer[GLOBAL_ID] += 0.5f*energy;
#endif
}

KERNEL void gridInterpolateForce(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
#ifdef USE_ESP
        , GLOBAL const real* RESTRICT espSpreadCoeffs, GLOBAL const real* RESTRICT espSpreadDerCoeffs
#endif
        ) {
    real3 data[SPREAD_ORDER];
    real3 ddata[SPREAD_ORDER];
#ifndef USE_ESP
    const real scale = RECIP((real) (PME_ORDER-1));
#endif

    // Process the atoms in spatially sorted order.  This improves cache performance when loading
    // the grid values.

    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        int atom = pmeAtomGridIndex[i].x;
        real3 force = make_real3(0);
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        real q = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        real q = CHARGE*EPSILON_FACTOR;
#endif
        if (q == 0)
            continue;

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
#ifdef USE_ESP
        for (int j = 0; j < ESP_ORDER; j++)
            data[j] = espHorner3(espSpreadCoeffs, j, ESP_ORDER, ESP_POLY_ORDER, dr);
        for (int j = 0; j < ESP_ORDER; j++)
            ddata[j] = espHorner3(espSpreadDerCoeffs, j, ESP_ORDER, ESP_DER_POLY_ORDER, dr);
#else
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        ddata[0] = -data[0];
        for (int j = 1; j < PME_ORDER; j++)
            ddata[j] = data[j-1]-data[j];
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
#endif

        // Compute the force on this atom.

        for (int ix = 0; ix < SPREAD_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;

            for (int iy = 0; iy < SPREAD_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;

                for (int iz = 0; iz < SPREAD_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real gridvalue = pmeGrid[index];
                    force.x += ddx*dy*data[iz].z*gridvalue;
                    force.y += dx*ddy*data[iz].z*gridvalue;
                    force.z += dx*dy*ddata[iz].z*gridvalue;
                }
            }
        }
        real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
        real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
        real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
#ifdef USE_PME_STREAM
        ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(forceX));
        ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceY));
        ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceZ));
#else
        forceBuffers[atom] += (mm_ulong) realToFixedPoint(forceX);
        forceBuffers[atom+PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceY);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(forceZ);
#endif
    }
}

KERNEL void gridInterpolateChargeDerivatives(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT derivatives, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int* RESTRICT atomIndices
        ) {
    real3 data[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));

    for (int i = GLOBAL_ID; i < NUM_INDICES; i += GLOBAL_SIZE) {
        int atom = atomIndices[i];
        real derivative = 0;
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.

        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP((real) (j-1));
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];

        // Compute the charge derivative on this atom.

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    derivative += dx*dy*data[iz].z*pmeGrid[ybase + zindex];
                }
            }
        }
        derivative *= EPSILON_FACTOR;
#ifdef USE_PME_STREAM
        ATOMIC_ADD(&derivatives[i], (mm_ulong) realToFixedPoint(derivative));
#else
        derivatives[i] += (mm_ulong) realToFixedPoint(derivative);
#endif
    }
}

KERNEL void addForces(GLOBAL const real4* RESTRICT forces, GLOBAL mm_long* RESTRICT forceBuffers) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real4 f = forces[atom];
        forceBuffers[atom] += realToFixedPoint(f.x);
        forceBuffers[atom+PADDED_NUM_ATOMS] += realToFixedPoint(f.y);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += realToFixedPoint(f.z);
    }
}

KERNEL void addEnergy(GLOBAL const mixed* RESTRICT pmeEnergyBuffer, GLOBAL mixed* RESTRICT energyBuffer, int bufferSize) {
    for (int i = GLOBAL_ID; i < bufferSize; i += GLOBAL_SIZE)
        energyBuffer[i] += pmeEnergyBuffer[i];
}
