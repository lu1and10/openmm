/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaTests.h"
#include "TestNonbondedForce.h"
#include <cuda.h>
#include <cstdlib>
#include <string>

namespace {

void setEspOutputTileEnv(const char* value) {
#ifdef WIN32
    _putenv_s("OPENMM_NONBONDED_ESP_OUTPUT_TILE_SPREAD", value == NULL ? "" : value);
#else
    if (value == NULL)
        unsetenv("OPENMM_NONBONDED_ESP_OUTPUT_TILE_SPREAD");
    else
        setenv("OPENMM_NONBONDED_ESP_OUTPUT_TILE_SPREAD", value, 1);
#endif
}

class ScopedEspOutputTileEnv {
public:
    ScopedEspOutputTileEnv(const char* value) {
        const char* old = getenv("OPENMM_NONBONDED_ESP_OUTPUT_TILE_SPREAD");
        hasOldValue = (old != NULL);
        if (hasOldValue)
            oldValue = old;
        setEspOutputTileEnv(value);
    }
    ~ScopedEspOutputTileEnv() {
        setEspOutputTileEnv(hasOldValue ? oldValue.c_str() : NULL);
    }
private:
    bool hasOldValue;
    std::string oldValue;
};

}

void testDeterministicForces() {
    // Check that the CudaDeterministicForces property works correctly.
    
    const int numParticles = 1000;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(2.1, 6, 0), Vec3(-1.5, -0.5, 6));
    NonbondedForce *nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    system.addForce(nonbonded);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 1 : -1, 1, 0);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*6);
    }
    VerletIntegrator integrator(0.001);
    map<string, string> properties;
    properties[CudaPlatform::CudaDeterministicForces()] = "true";
    Context context(system, integrator, platform, properties);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces);
    State state2 = context.getState(State::Forces);
    
    // All forces should be *exactly* equal.
    
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL(state1.getForces()[i][0], state2.getForces()[i][0]);
        ASSERT_EQUAL(state1.getForces()[i][1], state2.getForces()[i][1]);
        ASSERT_EQUAL(state1.getForces()[i][2], state2.getForces()[i][2]);
    }
}

bool canRunHugeTest() {
    // Create a minimal context just to see which device is being used.

    System system;
    system.addParticle(1.0);
    VerletIntegrator integrator(1.0);
    Context context(system, integrator, platform);
    int deviceIndex = stoi(platform.getPropertyValue(context, CudaPlatform::CudaDeviceIndex()));

    // Find out how much memory the device has.

    CUdevice device;
    cuDeviceGet(&device, deviceIndex);
    size_t memory;
    cuDeviceTotalMem(&memory, device);

    // Only run the huge test if the device has at least 4 GB of memory.

    return (memory >= 4L*(1<<30));
}

void testEspUsesFormulaSelectedStencilOrder() {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(2.1, 0, 0), Vec3(0, 2.1, 0), Vec3(0, 0, 2.1));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setReciprocalSpaceKernelType(NonbondedForce::ESPKernel);
    nonbonded->setCutoffDistance(1.0);
    nonbonded->setEwaldErrorTolerance(1e-4);
    system.addForce(nonbonded);
    for (int i = 0; i < 2; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i == 0 ? 1.0 : -1.0, 1.0, 0.0);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(1.0, 0, 0)});

    double alpha;
    int nx, ny, nz;
    nonbonded->getPMEParametersInContext(context, alpha, nx, ny, nz);
    ASSERT_EQUAL(10, nx);
    ASSERT_EQUAL(10, ny);
    ASSERT_EQUAL(10, nz);
}

void testEspRequiresPme() {
    System system;
    system.addParticle(1.0);
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    nonbonded->setReciprocalSpaceKernelType(NonbondedForce::ESPKernel);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.001);
    bool threw = false;
    try {
        Context context(system, integrator, platform);
    }
    catch (const OpenMMException& e) {
        ASSERT(string(e.what()).find("only supported for PME") != string::npos);
        threw = true;
    }
    ASSERT(threw);
}

void testEspDirectSpaceUsesPswfSplit() {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setReciprocalSpaceKernelType(NonbondedForce::ESPKernel);
    nonbonded->setReciprocalSpaceForceGroup(1);
    nonbonded->setCutoffDistance(0.5);
    nonbonded->setEwaldErrorTolerance(1e-4);
    system.addForce(nonbonded);
    system.addParticle(1.0);
    system.addParticle(1.0);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(0.4, 0, 0)});

    State state = context.getState(State::Forces | State::Energy, true, 1<<0);
    const double directEnergy = state.getPotentialEnergy();
    const double espShortRangeScale = 1.847619872734e-3;
    const double espShortRangeForceScale = 2.817916144554e-2;
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*espShortRangeScale/0.4, directEnergy, 2e-5);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*espShortRangeForceScale/(0.4*0.4), fabs(state.getForces()[0][0]), 2e-3);
    ASSERT_EQUAL_TOL(0.0, state.getForces()[0][1], 1e-5);
    ASSERT_EQUAL_TOL(0.0, state.getForces()[0][2], 1e-5);
}

void testEspExcludedPairBeyondCutoff() {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setReciprocalSpaceKernelType(NonbondedForce::ESPKernel);
    nonbonded->setReciprocalSpaceForceGroup(1);
    nonbonded->setCutoffDistance(1.0);
    nonbonded->setEwaldErrorTolerance(1e-5);
    system.addForce(nonbonded);
    system.addParticle(1.0);
    system.addParticle(1.0);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(-1.0, 1.0, 0.0);
    nonbonded->addException(0, 1, 0.0, 1.0, 0.0);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    const double r = 1.2;
    context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    State state = context.getState(State::Forces | State::Energy, true, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0/r, state.getPotentialEnergy(), 1e-4);
    ASSERT_EQUAL_TOL(-ONE_4PI_EPS0/(r*r), state.getForces()[0][0], 1e-4);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0/(r*r), state.getForces()[1][0], 1e-4);
    ASSERT_EQUAL_TOL(0.0, state.getForces()[0][1], 1e-5);
    ASSERT_EQUAL_TOL(0.0, state.getForces()[0][2], 1e-5);
}

void testEspForcedOutputTileFallsBackWhenScratchIsTooLarge() {
    ScopedEspOutputTileEnv env("1");
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setReciprocalSpaceKernelType(NonbondedForce::ESPKernel);
    nonbonded->setCutoffDistance(0.9);
    nonbonded->setEwaldErrorTolerance(1e-6);
    nonbonded->setUseDispersionCorrection(false);
    system.addForce(nonbonded);
    system.addParticle(1.0);
    system.addParticle(1.0);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(-1.0, 1.0, 0.0);

    VerletIntegrator integrator(0.001);
    map<string, string> properties;
    properties["Precision"] = "double";
    Context context(system, integrator, platform, properties);
    context.setPositions({Vec3(0, 0, 0), Vec3(0.4, 0, 0)});
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL(2, state.getForces().size());
}

void computePmeKernelForces(NonbondedForce::ReciprocalSpaceKernelType kernel, bool useChargeOffsets, double tolerance,
        double& energy, vector<Vec3>& forces) {
    const int cells = 6;
    const double spacing = 0.31;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(cells*spacing, 0, 0), Vec3(0, cells*spacing, 0), Vec3(0, 0, cells*spacing));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setReciprocalSpaceKernelType(kernel);
    nonbonded->setCutoffDistance(0.9);
    nonbonded->setEwaldErrorTolerance(tolerance);
    system.addForce(nonbonded);
    vector<Vec3> positions;
    for (int x = 0; x < cells; x++) {
        for (int y = 0; y < cells; y++) {
            for (int z = 0; z < cells; z++) {
                int base = system.getNumParticles();
                Vec3 o((x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing);
                Vec3 a(0.09572, 0, 0);
                Vec3 b(-0.02399, 0.09266, 0);
                if ((x+y+z)%3 == 1) {
                    a = Vec3(0, 0.09572, 0);
                    b = Vec3(0.09266, -0.02399, 0);
                }
                else if ((x+y+z)%3 == 2) {
                    a = Vec3(0, 0, 0.09572);
                    b = Vec3(0, 0.09266, -0.02399);
                }
                if ((x+y+z)%2 == 1) {
                    a *= -1;
                    b *= -1;
                }
                system.addParticle(16.0);
                system.addParticle(1.0);
                system.addParticle(1.0);
                nonbonded->addParticle(-0.834, 1.0, 0.0);
                nonbonded->addParticle(0.417, 1.0, 0.0);
                nonbonded->addParticle(0.417, 1.0, 0.0);
                nonbonded->addException(base, base+1, 0.0, 1.0, 0.0);
                nonbonded->addException(base, base+2, 0.0, 1.0, 0.0);
                nonbonded->addException(base+1, base+2, 0.0, 1.0, 0.0);
                positions.push_back(o);
                positions.push_back(o+a);
                positions.push_back(o+b);
            }
        }
    }
    if (useChargeOffsets) {
        nonbonded->addGlobalParameter("scale", 0.3);
        nonbonded->addParticleParameterOffset("scale", 0, 0.1, 0.0, 0.0);
        nonbonded->addParticleParameterOffset("scale", 1, -0.05, 0.0, 0.0);
        nonbonded->addParticleParameterOffset("scale", 2, -0.05, 0.0, 0.0);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    if (useChargeOffsets)
        context.setParameter("scale", 0.7);
    State state = context.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    forces = state.getForces();
}

void assertEspMatchesPmeCoulomb(bool useChargeOffsets) {
    double pmeEnergy, espEnergy;
    vector<Vec3> pmeForces, espForces;
    computePmeKernelForces(NonbondedForce::PMEKernel, useChargeOffsets, 5e-7, pmeEnergy, pmeForces);
    computePmeKernelForces(NonbondedForce::ESPKernel, useChargeOffsets, 1e-5, espEnergy, espForces);

    double forceNorm2 = 0.0;
    double forceDiff2 = 0.0;
    for (int i = 0; i < pmeForces.size(); i++) {
        Vec3 diff = espForces[i]-pmeForces[i];
        forceNorm2 += pmeForces[i].dot(pmeForces[i]);
        forceDiff2 += diff.dot(diff);
    }
    const double forceRelError = sqrt(forceDiff2/forceNorm2);
    const double energyRelError = fabs(espEnergy-pmeEnergy)/fabs(pmeEnergy);
    ASSERT(forceRelError < 1.5e-5);
    ASSERT(energyRelError < 1.5e-5);
}

void testEspMatchesPmeCoulomb() {
    assertEspMatchesPmeCoulomb(false);
}

void testEspMatchesPmeCoulombWithChargeOffsets() {
    assertEspMatchesPmeCoulomb(true);
}

void runPlatformTests() {
    testParallelComputation(NonbondedForce::NoCutoff);
    testParallelComputation(NonbondedForce::Ewald);
    testParallelComputation(NonbondedForce::PME);
    testParallelComputation(NonbondedForce::LJPME);
    testEspUsesFormulaSelectedStencilOrder();
    testEspRequiresPme();
    testEspDirectSpaceUsesPswfSplit();
    testEspExcludedPairBeyondCutoff();
    testEspForcedOutputTileFallsBackWhenScratchIsTooLarge();
    testEspMatchesPmeCoulomb();
    testEspMatchesPmeCoulombWithChargeOffsets();
    testReordering();
    testDeterministicForces();
    if (canRunHugeTest())
        testHugeSystem();
}
