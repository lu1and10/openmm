const float4 exclusionParams = PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
const real r = SQRT(r2);
#if !USE_ESP || DO_LJPME
const real invR = RECIP(r);
#endif
#if USE_ESP
real tempForce = 0.0f;
real scaledR = r*ESP_INV_CUTOFF;
real nonzero = (scaledR > (real) 1e-6f ? (real) 1 : (real) 0);
real insideCutoff = (scaledR < (real) 1 ? (real) 1 : (real) 0);
real scaledEvalR = (scaledR > (real) 1e-6f ? scaledR : (real) 1e-6f);
scaledEvalR = (scaledEvalR < (real) 1 ? scaledEvalR : (real) 1);
real scaledT = 2.0f*scaledEvalR-1.0f;
real polyEnergyScale = ESP_LONG_RANGE_ENERGY_POLY;
real polyForceScale = ESP_LONG_RANGE_FORCE_POLY;
real longRangeEnergyScale = insideCutoff*polyEnergyScale + ((real) 1-insideCutoff);
real longRangeForceScale = insideCutoff*polyForceScale + ((real) 1-insideCutoff);
real minR = (real) 1e-6f/ESP_INV_CUTOFF;
real safeR = (r > minR ? r : minR);
real safeInvR = RECIP(safeR);
const real prefactor = exclusionParams.x*safeInvR;
tempForce = -nonzero*prefactor*longRangeForceScale;
energy -= nonzero*prefactor*longRangeEnergyScale +
        ((real) 1-nonzero)*ESP_LONG_RANGE_ENERGY_LIMIT*exclusionParams.x;
#else
const real alphaR = EWALD_ALPHA*r;
const real expAlphaRSqr = EXP(-alphaR*alphaR);
real tempForce = 0.0f;
if (alphaR > 1e-6f) {
    const real erfAlphaR = ERF(alphaR);
    const real prefactor = exclusionParams.x*invR;
    tempForce = -prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    energy -= prefactor*erfAlphaR;
}
else {
    energy -= TWO_OVER_SQRT_PI*EWALD_ALPHA*exclusionParams.x;
}
#endif
#if DO_LJPME
const real dispersionAlphaR = EWALD_DISPERSION_ALPHA*r;
const real dar2 = dispersionAlphaR*dispersionAlphaR;
const real dar4 = dar2*dar2;
const real dar6 = dar4*dar2;
const real invR2 = invR*invR;
const real expDar2 = EXP(-dar2);
const real c6 = 64*exclusionParams.y*exclusionParams.y*exclusionParams.y*exclusionParams.z;
const real coef = invR2*invR2*invR2*c6;
const real eprefac = 1.0f + dar2 + 0.5f*dar4;
const real dprefac = eprefac + dar6/6.0f;
energy += coef*(1.0f - expDar2*eprefac);
tempForce += 6.0f*coef*(1.0f - expDar2*dprefac);
#endif
#if USE_ESP
delta *= tempForce*safeInvR*safeInvR;
#else
if (r > 0)
    delta *= tempForce*invR*invR;
#endif
real3 force1 = -delta;
real3 force2 = delta;
