/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
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

// Prolate spheroidal wave function coefficient builder for ESP.  This runs at
// Context initialization; GPU kernels only use the generated polynomial tables.

#ifndef OPENMM_ESP_PSWF_H_
#define OPENMM_ESP_PSWF_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pswf {

// Legendre polynomial helpers adapted from the ESP CPU implementation.

static inline void legepol(double x, int n, double& pol, double& der) {
    double pkm1 = 1.0;
    double pk = x;
    double pkp1;

    if (n == 0) {
        pol = 1.0;
        der = 0.0;
        return;
    }

    if (n == 1) {
        pol = x;
        der = 1.0;
        return;
    }

    pk = 1.0;
    pkp1 = x;

    for (int k = 1; k < n; ++k) {
        pkm1 = pk;
        pk = pkp1;
        pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
    }

    pol = pkp1;
    der = n * (x * pkp1 - pk) / (x * x - 1);
}

static inline void legetayl(double pol, double der, double x, double h,
                             int n, int k, double& sum, double& sumder) {
    double done = 1.0;
    double q0 = pol;
    double q1 = der * h;
    double q2 = (2 * x * der - n * (n + done) * pol) / (1 - x * x);
    q2 = q2 * h * h / 2;

    sum = q0 + q1 + q2;
    sumder = q1 / h + q2 * 2 / h;

    if (k <= 2) return;

    double qi = q1;
    double qip1 = q2;

    for (int i = 1; i <= k - 2; ++i) {
        double d = 2 * x * (i + 1) * (i + 1) / h * qip1 - (n * (n + done) - i * (i + 1)) * qi;
        d = d / (i + 1) / (i + 2) * h * h / (1 - x * x);
        double qip2 = d;

        sum += qip2;
        sumder += d * (i + 2) / h;

        qi = qip1;
        qip1 = qip2;
    }
}

// Gauss-Legendre quadrature: itype=1 computes both roots and weights.
static inline void legerts(int itype, int n, double* ts, double* whts) {
    int k = 30;
    double d = 1.0;
    double d2 = d + 1.0e-24;
    if (d2 != d) {
        k = 54;
    }

    int half = n / 2;
    int ifodd = n - 2 * half;
    double pi_val = atan(1.0) * 4.0;
    double h = pi_val / (2.0 * n);

    int ii = 0;
    for (int i = 1; i <= n; i++) {
        if (i < (n / 2 + 1)) {
            continue;
        }
        ii++;
        double t = (2.0 * i - 1.0) * h;
        ts[ii - 1] = -cos(t);
    }

    double pol = 1.0, der = 0.0;
    double x0 = 0.0;
    legepol(x0, n, pol, der);
    double x1 = ts[0];

    int n2 = (n + 1) / 2;
    double pol3 = pol, der3 = der;

    for (int kk = 1; kk <= n2; kk++) {
        if ((ifodd == 1) && (kk == 1)) {
            ts[kk - 1] = x0;
            if (itype > 0) {
                whts[kk - 1] = der;
            }
            x0 = x1;
            x1 = ts[kk];
            pol3 = pol;
            der3 = der;
            continue;
        }

        int ifstop = 0;
        for (int i = 1; i <= 10; i++) {
            double hh = x1 - x0;

            legetayl(pol3, der3, x0, hh, n, k, pol, der);
            x1 = x1 - pol / der;

            if (fabs(pol) < 1.0e-12) {
                ifstop++;
            }
            if (ifstop == 3) {
                break;
            }
        }

        ts[kk - 1] = x1;
        if (itype > 0) {
            whts[kk - 1] = der;
        }

        x0 = x1;
        x1 = ts[kk];
        pol3 = pol;
        der3 = der;
    }

    for (int i = n2; i >= 1; i--) {
        ts[i - 1 + half] = ts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        ts[i - 1] = -ts[n - i];
    }
    if (itype <= 0) {
        return;
    }

    for (int i = n2; i >= 1; i--) {
        whts[i - 1 + half] = whts[i - 1];
    }
    for (int i = 1; i <= half; i++) {
        whts[i - 1] = whts[n - i];
    }

    for (int i = 0; i < n; i++) {
        double tmp = 1.0 - ts[i] * ts[i];
        whts[i] = 2.0 / tmp / (whts[i] * whts[i]);
    }
}

// Convenience: compute Gauss-Legendre nodes and weights.
static inline void legerts(int n, double* ts, double* whts) {
    legerts(1, n, ts, whts);
}

static inline void prolcoef(double lambda, int k, double c, double& alpha, double& beta, double& gamma) {
    const double kf = k;
    const double alpha0 = kf*(kf-1.0)/((2.0*kf+1.0)*(2.0*kf-1.0));
    const double beta0 = ((kf+1.0)*(kf+1.0)/(2.0*kf+3.0) + kf*kf/(2.0*kf-1.0))/(2.0*kf+1.0);
    const double gamma0 = (kf+1.0)*(kf+2.0)/((2.0*kf+1.0)*(2.0*kf+3.0));
    alpha = -c*c*alpha0;
    beta = lambda - kf*(kf+1.0) - c*c*beta0;
    gamma = -c*c*gamma0;
}

static inline void prolmatr(std::vector<double>& as, std::vector<double>& bs,
                            std::vector<double>& cs, int n, double c, double lambda) {
    for (int i = 0; 2*i <= n+2; ++i) {
        prolcoef(lambda, 2*i, c, as[i], bs[i], cs[i]);
        if (i != 0)
            as[i] *= std::sqrt((2.0*i+0.5)/(2.0*i-1.5));
        cs[i] *= std::sqrt((2.0*i+0.5)/(2.0*i+2.5));
    }
}

static inline void prolql1(int n, std::vector<double>& diagonal, std::vector<double>& offDiagonal) {
    if (n == 1)
        return;
    for (int i = 1; i < n; ++i)
        offDiagonal[i-1] = offDiagonal[i];
    offDiagonal[n-1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;
        while (true) {
            int m = l;
            for (; m < n-1; ++m) {
                const double scale = std::abs(diagonal[m]) + std::abs(diagonal[m+1]);
                if (scale+std::abs(offDiagonal[m]) == scale)
                    break;
            }
            if (m == l)
                break;
            if (iter++ == 30)
                throw std::runtime_error("Pswf0: tridiagonal eigensolver failed to converge");

            double g = (diagonal[l+1]-diagonal[l])/(2.0*offDiagonal[l]);
            double r = std::hypot(g, 1.0);
            g = diagonal[m]-diagonal[l]+offDiagonal[l]/(g+std::copysign(r, g));

            double sine = 1.0;
            double cosine = 1.0;
            double p = 0.0;
            for (int i = m-1; i >= l; --i) {
                const double f = sine*offDiagonal[i];
                const double b = cosine*offDiagonal[i];
                r = std::hypot(f, g);
                offDiagonal[i+1] = r;
                if (r == 0.0) {
                    diagonal[i+1] -= p;
                    offDiagonal[m] = 0.0;
                    break;
                }
                sine = f/r;
                cosine = g/r;
                g = diagonal[i+1]-p;
                r = (diagonal[i]-g)*sine + 2.0*cosine*b;
                p = sine*r;
                diagonal[i+1] = g+p;
                g = cosine*r-b;
            }
            if (r == 0.0)
                break;
            diagonal[l] -= p;
            offDiagonal[l] = g;
            offDiagonal[m] = 0.0;
        }

        for (int i = l; i > 0 && diagonal[i] < diagonal[i-1]; --i)
            std::swap(diagonal[i], diagonal[i-1]);
    }
}

static inline void prolfact(std::vector<double>& diagonal, const std::vector<double>& upper,
                            const std::vector<double>& lower, int n,
                            std::vector<double>& down, std::vector<double>& up,
                            std::vector<double>& inverseDiagonal) {
    for (int i = 0; i+1 < n; ++i) {
        const double factor = lower[i+1]/diagonal[i];
        diagonal[i+1] -= upper[i]*factor;
        down[i] = factor;
        up[i+1] = upper[i]/diagonal[i+1];
        inverseDiagonal[i+1] = 1.0/diagonal[i+1];
    }
    inverseDiagonal[0] = 1.0/diagonal[0];
}

static inline void prolsolv(const std::vector<double>& down, const std::vector<double>& up,
                            const std::vector<double>& inverseDiagonal, int n,
                            std::vector<double>& rhs) {
    for (int i = 0; i+1 < n; ++i)
        rhs[i+1] -= down[i]*rhs[i];
    for (int i = n-1; i > 0; --i) {
        rhs[i-1] -= rhs[i]*up[i];
        rhs[i] *= inverseDiagonal[i];
    }
    rhs[0] *= inverseDiagonal[0];
}

static inline void prolfun0(int n, double c, std::vector<double>& coefficients, double eps) {
    // Shift the selected eigenvalue slightly before inverse iteration.  This is
    // the same delta used by the original PSWF setup and avoids factoring an
    // exactly singular tridiagonal system.
    const double eigenvalueShift = 1.0e-8;
    const int numIterations = 4;
    const int dimension = n/2;
    coefficients.assign(dimension+3, 1.0);

    std::vector<double> as(dimension+2), bs(dimension+2), cs(dimension+2);
    std::vector<double> down(dimension+2), up(dimension+2), inverseDiagonal(dimension+2);

    prolmatr(as, bs, cs, n, c, 0.0);
    prolql1(dimension, bs, as);

    const double lambda = -bs[dimension-1] + eigenvalueShift;
    prolmatr(as, bs, cs, n, c, lambda);
    prolfact(bs, cs, as, dimension, down, up, inverseDiagonal);

    for (int iter = 0; iter < numIterations; ++iter) {
        prolsolv(down, up, inverseDiagonal, dimension, coefficients);
        double norm = 0.0;
        for (int j = 0; j < dimension; ++j)
            norm += coefficients[j]*coefficients[j];
        norm = std::sqrt(norm);
        for (int j = 0; j < dimension; ++j)
            coefficients[j] /= norm;
    }

    int lastSignificant = 0;
    for (int i = 0; i < dimension; ++i) {
        if (std::abs(coefficients[i]) > eps)
            lastSignificant = i;
        coefficients[i] *= std::sqrt(2.0*i+0.5);
    }
    coefficients.resize(lastSignificant+1);
}

static inline void prolps0i(double c, std::vector<double>& coefficients) {
    static const int expansionOrders[] = {
        48, 64, 80, 92, 106, 120, 130, 144, 156, 168,
        178, 190, 202, 214, 224, 236, 248, 258, 268, 280
    };
    const int bucket = static_cast<int>(c/10.0);
    const int n = (bucket < (int) (sizeof(expansionOrders)/sizeof(expansionOrders[0])) ?
                   expansionOrders[bucket] : static_cast<int>(c*3.0)/2);
    prolfun0(n, c, coefficients, 1.0e-16);
}

static inline double evaluateRaw(const std::vector<double>& coefficients,
                                  const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                                  double x) {
    const double xSquared = x*x;
    double pjm1 = 0.0;
    double pjm2 = 1.0;
    double value = coefficients[0];

    for (size_t i = 1; i < recurrenceCoefficients.size(); ++i) {
        const std::array<double, 3>& r = recurrenceCoefficients[i];
        const double p = pjm2*(xSquared*r[0]-r[1]) - pjm1*r[2];
        value += coefficients[i]*p;
        pjm1 = pjm2;
        pjm2 = p;
    }
    return value;
}

static inline double evaluateRawDerivative(const std::vector<double>& coefficients,
                                             const std::vector<std::array<double, 3>>& recurrenceCoefficients,
                                             double x) {
    const double xSquared = x*x;
    double pjm1 = 0.0;
    double pjm2 = 1.0;
    double dPjm1 = 0.0;
    double dPjm2 = 0.0;
    double dValue = 0.0;

    for (size_t i = 1; i < recurrenceCoefficients.size(); ++i) {
        const std::array<double, 3>& r = recurrenceCoefficients[i];
        const double a = xSquared*r[0]-r[1];
        const double dA = 2.0*x*r[0];
        const double p = pjm2*a - pjm1*r[2];
        const double dP = dPjm2*a + pjm2*dA - dPjm1*r[2];
        dValue += coefficients[i]*dP;
        pjm1 = pjm2;
        pjm2 = p;
        dPjm1 = dPjm2;
        dPjm2 = dP;
    }
    return dValue;
}

class Pswf0 {
public:
    explicit Pswf0(double c) {
        if (c <= 0.0 || c > 30.0)
            throw std::invalid_argument("Pswf0: c must be in (0, 30]");

        // PSWF_0 is even, so only P_0, P_2, P_4, ... coefficients are stored.
        // The eigenvector normalization is intentionally left unchanged; callers
        // use ratios, integrals, or matching Fourier factors where the scale is
        // handled explicitly.
        prolps0i(c, legendreCoefficients);
        recurrenceCoefficients.resize(legendreCoefficients.size());
        for (size_t i = 1; i < recurrenceCoefficients.size(); ++i) {
            const double ell = 2.0*i - 1.0;
            recurrenceCoefficients[i][0] = ((2.0*ell-1.0)*(2.0*ell+1.0))/(ell*(ell+1.0));
            recurrenceCoefficients[i][1] =
                ((2.0*ell+1.0)*(ell-1.0)*(ell-1.0) + ell*ell*(2.0*ell-3.0))/
                (ell*(ell+1.0)*(2.0*ell-3.0));
            recurrenceCoefficients[i][2] =
                ((2.0*ell+1.0)*(ell-1.0)*(ell-2.0))/(ell*(ell+1.0)*(2.0*ell-3.0));
        }
    }

    double eval(double x) const {
        if (std::abs(x) > 1.0)
            return 0.0;
        return evaluateRaw(legendreCoefficients, recurrenceCoefficients, x);
    }

    double evalDerivative(double x) const {
        if (std::abs(x) > 1.0)
            return 0.0;
        return evaluateRawDerivative(legendreCoefficients, recurrenceCoefficients, x);
    }

    double evalIntegral(double upper) const {
        if (upper == 0.0)
            return 0.0;
        const double sign = (upper < 0.0 ? -1.0 : 1.0);
        const double limit = std::min(std::abs(upper), 1.0);
        const double half = 0.5*limit;

        const int npts = 200;
        std::vector<double> nodes(npts), weights(npts);
        legerts(npts, nodes.data(), weights.data());

        double sum = 0.0;
        for (int i = 0; i < npts; ++i)
            sum += weights[i]*eval(half*(nodes[i]+1.0));
        return sign*half*sum;
    }

private:
    std::vector<double> legendreCoefficients;
    std::vector<std::array<double, 3>> recurrenceCoefficients;
};

// ============================================================================
// Tolerance-to-c lookup table adapted from the ESP CPU implementation.
// ============================================================================

static inline void prolc180(double eps, double& c) {
    static const double cs[] = {
        0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01, 0.30035E+01, 0.33409E+01,
        0.36598E+01, 0.39658E+01, 0.42621E+01, 0.45513E+01, 0.48347E+01, 0.51136E+01, 0.53887E+01,
        0.56606E+01, 0.59299E+01, 0.61968E+01, 0.64616E+01, 0.67247E+01, 0.69862E+01, 0.72462E+01,
        0.75049E+01, 0.77625E+01, 0.80189E+01, 0.82744E+01, 0.85289E+01, 0.87826E+01, 0.90355E+01,
        0.92877E+01, 0.95392E+01, 0.97900E+01, 0.10040E+02, 0.10290E+02, 0.10539E+02, 0.10788E+02,
        0.11036E+02, 0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02, 0.12270E+02, 0.12516E+02,
        0.12762E+02, 0.13007E+02, 0.13251E+02, 0.13496E+02, 0.13740E+02, 0.13984E+02, 0.14228E+02,
        0.14471E+02, 0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02, 0.15685E+02, 0.15927E+02,
        0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02, 0.17135E+02, 0.17376E+02, 0.17617E+02,
        0.17858E+02, 0.18098E+02, 0.18339E+02, 0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02,
        0.19539E+02, 0.19778E+02, 0.20018E+02, 0.20257E+02, 0.20496E+02, 0.20736E+02, 0.20975E+02,
        0.21214E+02, 0.21452E+02, 0.21691E+02, 0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02,
        0.22884E+02, 0.23122E+02, 0.23360E+02, 0.23598E+02, 0.23836E+02, 0.24074E+02, 0.24311E+02,
        0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02, 0.25499E+02, 0.25737E+02, 0.25974E+02,
        0.26211E+02, 0.26448E+02, 0.26685E+02, 0.26922E+02, 0.27159E+02, 0.27396E+02, 0.27633E+02,
        0.27870E+02, 0.28106E+02, 0.28343E+02, 0.28580E+02, 0.28816E+02, 0.29053E+02, 0.29289E+02,
        0.29526E+02, 0.29762E+02, 0.29998E+02, 0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02,
        0.31179E+02, 0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02, 0.32594E+02,
        0.32830E+02, 0.33066E+02, 0.33301E+02, 0.33537E+02, 0.33773E+02, 0.34008E+02, 0.34244E+02,
        0.34479E+02, 0.34714E+02, 0.34950E+02, 0.35185E+02, 0.35421E+02, 0.35656E+02, 0.35891E+02,
        0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02, 0.37067E+02, 0.37302E+02, 0.37537E+02,
        0.37772E+02, 0.38007E+02, 0.38242E+02, 0.38477E+02, 0.38712E+02, 0.38947E+02, 0.39181E+02,
        0.39416E+02, 0.39651E+02, 0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02, 0.40824E+02,
        0.41059E+02, 0.41294E+02, 0.41528E+02, 0.41763E+02, 0.41997E+02, 0.42232E+02, 0.42466E+02,
        0.42700E+02, 0.42935E+02, 0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02, 0.44107E+02,
        0.44341E+02, 0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02};

    double e = eps;
    if (e < 1.0e-18) e = 1e-18;
    double d = -log10(e);
    int i = static_cast<int>(d * 10 + 0.1);
    c = cs[i - 1];
}

static inline double getProlateC(double tol) {
    double c;
    prolc180(tol, c);
    return c;
}

static inline int estimateEspOrder(double tol) {
    if (tol <= 0.0 || tol >= 1.0)
        throw std::invalid_argument("estimateEspOrder: tolerance must be in (0, 1)");
    double p = -std::log10(tol);
    double rounded = std::round(p);
    int order;
    if (std::abs(p-rounded) < 0.2)
        order = 2*(int) rounded - 2;
    else
        order = 2*(int) std::ceil(p) - 3;
    return std::min(std::max(order, 4), 12);
}

// ============================================================================
// Chebyshev nodes and interpolation for host-side polynomial order selection.
// ============================================================================

static const int MAX_CHEB_ORDER = 30;

static inline void chebNodes1D(int order, std::vector<double>& nodes, double a = 0, double b = 1) {
    const double pi = 3.1415926535897932384626433832795028841;
    nodes.resize(order);
    for (int i = 0; i < order; i++) {
        nodes[i] = -cos((i + 0.5) * pi / order) * 0.5 + 0.5;
        nodes[i] = nodes[i] * (b - a) + a;
    }
}

static inline void chebBasis1D(int order, const std::vector<double>& x, std::vector<double>& y,
                                  double a = 0, double b = 1) {
    int n = (int)x.size();
    y.resize(order * n);

    if (order > 0) {
        for (int i = 0; i < n; i++) {
            y[i] = 1.0;
        }
    }
    if (order > 1) {
        for (int i = 0; i < n; i++) {
            y[i + n] = x[i] * 2 / (b - a) - 2 * a / (b - a) - 1;
        }
    }
    for (int i = 2; i < order; i++) {
        for (int j = 0; j < n; j++) {
            y[i * n + j] = 2 * y[n + j] * y[i * n - n + j] - y[i * n - 2 * n + j];
        }
    }
}

static inline void chebInterp1D(int order, std::vector<double>& fn_v,
                                    std::vector<double>& coeff) {
    std::vector<double> x, p;
    chebNodes1D(order, x);
    chebBasis1D(order, x, p);

    const size_t dof = fn_v.size() / order;
    assert(fn_v.size() == dof * (size_t)order);
    coeff.resize(dof * (size_t)order);

    const double invOrder = 1.0 / (double)order;
    const double twoInvOrder = 2.0 * invOrder;
    for (size_t idof = 0; idof < dof; ++idof) {
        const size_t offset = idof * (size_t)order;
        for (int k = 0; k < order; ++k) {
            double sum = 0.0;
            const double* pk = &p[(size_t)k * (size_t)order];
            for (int j = 0; j < order; ++j) {
                sum += fn_v[offset + j] * pk[j];
            }
            coeff[offset + (size_t)k] = (k == 0) ? (sum * invOrder) : (sum * twoInvOrder);
        }
    }
}

// ============================================================================
// Monomial interpolation via Newton divided differences. This is adapted from
// the ESP CPU implementation, but stores coefficients in descending order so
// GPU kernels can evaluate them by forward Horner loops.
// ============================================================================

static inline void monomialInterp1D(int order, int nnodes, std::vector<double>& fn_v,
                                       std::vector<double>& coeff,
                                       double a = 0, double b = 1) {
    assert(order == nnodes);

    std::vector<double> x;
    chebNodes1D(nnodes, x, a, b);

    auto multiply_x = [](const std::vector<double>& p, double x_in) {
        std::vector<double> r(p.size() + 1, 0.0);
        for (size_t i = 0; i < p.size(); ++i) {
            r[i] += -x_in * p[i];
            r[i + 1] += p[i];
        }
        return r;
    };

    const size_t dof = fn_v.size() / nnodes;
    assert(fn_v.size() == dof * (size_t)nnodes);

    std::vector<double> newton_coeffs = fn_v;
    coeff.assign(dof * (size_t)order, 0.0);

    for (size_t idof = 0; idof < dof; ++idof) {
        const size_t fn_offset = idof * (size_t)nnodes;
        const size_t coeff_offset = idof * (size_t)order;

        for (int j = 1; j < nnodes; ++j) {
            for (int i = nnodes - 1; i >= j; --i) {
                newton_coeffs[fn_offset + i] =
                    (newton_coeffs[fn_offset + i] - newton_coeffs[fn_offset + i - 1]) / (x[i] - x[i - j]);
            }
        }

        coeff[coeff_offset + (size_t)(order - 1)] = newton_coeffs[fn_offset];
        std::vector<double> basis{1.0};
        for (int j = 1; j < nnodes; ++j) {
            basis = multiply_x(basis, x[j - 1]);
            const double newton_coeff = newton_coeffs[fn_offset + j];
            for (size_t m = 0; m < basis.size(); ++m) {
                coeff[coeff_offset + (size_t)(order - 1) - m] += newton_coeff * basis[m];
            }
        }
    }
}

struct EspCoefficients {
    double splitC;         // Splitting bandwidth (larger, controls Fourier truncation)
    double windowC;        // Window bandwidth (smaller, controls aliasing; windowC = pi/2 * order)
    double splitIntegral;  // Integral: int_0^1 psi_s(x) dx
    double splitValueAt0;  // psi_s(0)
    double splitLambda;    // Fourier eigenvalue for splitting PSWF
    int order;             // Stencil size

    // Spreading uses windowC (monomial, descending Horner order)
    int spreadPolyOrder;
    std::vector<float> spreadCoeffs;       // [spreadPolyOrder * order]
    std::vector<float> spreadDerCoeffs;    // [spreadDerPolyOrder * order]
    int spreadDerPolyOrder;

    // Fourier splitting uses splitC (monomial, descending Horner order)
    std::vector<double> splitFourierCoeffs;    // Monomial coefficients for Horner evaluation

    // Direct-space split scales use splitC (monomial, descending Horner order)
    std::vector<double> longRangeEnergyCoeffs;     // Monomial coefficients for Horner evaluation
    std::vector<double> longRangeForceCoeffs;      // Monomial coefficients for Horner evaluation

    double windowLambda;    // Fourier eigenvalue for window
};

static inline double evaluateMonomialDescending(const std::vector<double>& coeffs, double x) {
    double value = 0.0;
    for (double c : coeffs)
        value = value*x + c;
    return value;
}

template <class Function>
static inline std::vector<double> fitMonomialAdaptive1D(double tol, int maxOrder, Function function) {
    if (tol <= 0.0 || tol >= 1.0)
        throw std::invalid_argument("fitMonomialAdaptive1D: tolerance must be in (0, 1)");
    if (maxOrder < 2)
        throw std::invalid_argument("fitMonomialAdaptive1D: maxOrder must be at least 2");

    // Fit in the centered coordinate t = 2*x-1.  This keeps the monomial basis
    // better conditioned than fitting powers of x on [0, 1].
    const int order = std::min(std::max(maxOrder, 2), 64);
    std::vector<double> nodes;
    chebNodes1D(order, nodes, -1, 1);
    std::vector<double> values(order);
    for (int i = 0; i < order; i++)
        values[i] = function(0.5*(nodes[i]+1.0));

    std::vector<double> chebCoeff;
    chebInterp1D(order, values, chebCoeff);
    double maxCoeff = 0.0;
    for (double c : chebCoeff)
        maxCoeff = std::max(maxCoeff, std::abs(c));

    int firstOrder = 2;
    if (maxCoeff > 0.0) {
        for (int i = 0; i < order; i++)
            if (std::abs(chebCoeff[i]) > 0.1*tol*maxCoeff)
                firstOrder = std::max(firstOrder, i+1);
    }
    firstOrder = std::min(firstOrder, order);

    const int numCheckPoints = 1025;
    for (int candidate = firstOrder; candidate <= order; candidate++) {
        chebNodes1D(candidate, nodes, -1, 1);
        values.resize(candidate);
        for (int i = 0; i < candidate; i++)
            values[i] = function(0.5*(nodes[i]+1.0));

        std::vector<double> coeffs;
        monomialInterp1D(candidate, candidate, values, coeffs, -1, 1);

        double maxAbsError = 0.0;
        for (int i = 0; i < numCheckPoints; i++) {
            const double x = (double) i/(numCheckPoints-1);
            const double t = 2.0*x-1.0;
            maxAbsError = std::max(maxAbsError, std::abs(evaluateMonomialDescending(coeffs, t)-function(x)));
        }
        if (maxAbsError <= tol) {
            return coeffs;
        }
    }
    throw std::runtime_error("fitMonomialAdaptive1D: failed to reach requested tolerance");
}

// The tolerance factors are setup-time knobs for direct-space polynomial
// order selection. They do not affect the reciprocal split or spread fits.
static inline void fitLongRangeSplitMonomials(double tol, int maxOrder, const Pswf0& pfun, double splitIntegral,
                                              std::vector<double>& energyCoeffs, std::vector<double>& forceCoeffs,
                                              double energyTolFactor = 0.001, double forceTolFactor = 0.1) {
    if (tol <= 0.0 || tol >= 1.0)
        throw std::invalid_argument("fitLongRangeSplitMonomials: tolerance must be in (0, 1)");
    if (maxOrder < 2)
        throw std::invalid_argument("fitLongRangeSplitMonomials: maxOrder must be at least 2");

    auto longRangeEnergy = [&](double x) {
        return pfun.evalIntegral(x)/splitIntegral;
    };
    auto longRangeForce = [&](double x) {
        return longRangeEnergy(x) - x*pfun.eval(x)/splitIntegral;
    };
    energyCoeffs = fitMonomialAdaptive1D(energyTolFactor*tol, maxOrder, longRangeEnergy);
    forceCoeffs = fitMonomialAdaptive1D(forceTolFactor*tol, maxOrder, longRangeForce);
}

// Helper: compute lambda (Fourier eigenvalue) for a given prolate function
static inline double computeLambda(const Pswf0& pfun, double c) {
    int quad_npts = 200;
    std::vector<double> xs(quad_npts), ws(quad_npts);
    legerts(quad_npts, xs.data(), ws.data());
    double lam = 0.0;
    for (int i = 0; i < quad_npts; i++) {
        lam += ws[i] * pfun.eval(xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    lam /= pfun.eval(0.5);
    return lam;
}

// Build ESP polynomial coefficients with separate splitting and window parameters.
// Per Bostrom, Tornberg, af Klinteberg (arXiv:2602.16591):
//   splitC controls the Ewald split (Fourier truncation error ~ e^{-splitC})
//   windowC controls the window/spreading (aliasing error ~ e^{-windowC}), windowC = pi/2 * order
// Splitting coefficients (splitFourierCoeffs, self-energy) use splitC.
// Spreading coefficients (spreadCoeffs, deconvolution) use windowC.
static inline EspCoefficients buildEspCoefficients(double splitC, double windowC, int espOrder,
                                                            double tol,
                                                            int max_poly_order = 16,
                                                            double directEnergyTolFactor = 0.001,
                                                            double directForceTolFactor = 0.1,
                                                            double spreadTolFactor = 1.0,
                                                            double spreadDerTolFactor = 1.0) {
    EspCoefficients esp;
    esp.splitC = splitC;
    esp.windowC = windowC;
    esp.order = espOrder;

    // --- Splitting PSWF for Fourier kernel and self-energy ---
    Pswf0 pfun_s(splitC);
    esp.splitIntegral = pfun_s.evalIntegral(1.0);
    esp.splitValueAt0 = pfun_s.eval(0.0);
    esp.splitLambda = computeLambda(pfun_s, splitC);

    // --- Window PSWF for spreading/interpolation ---
    Pswf0 pfun_w(windowC);
    esp.windowLambda = computeLambda(pfun_w, windowC);

    // Polynomial order selection uses Chebyshev coefficient decay filtering,
    // then converts the selected approximants to monomial coefficients for
    // runtime Horner evaluation.
    const double tol_coeff_spread  = spreadTolFactor*tol;
    const double tol_coeff_spread_der = spreadDerTolFactor*tol;
    const double tol_coeff_fourier = tol;

    // --- Spreading coefficients use windowC (window PSWF) ---
    {
        int order = MAX_CHEB_ORDER;
        std::vector<double> nodes;
        chebNodes1D(order, nodes);

        int dof = espOrder;
        std::vector<double> fn_v(dof * order);
        for (int idof = 0; idof < dof; idof++) {
            for (int i = 0; i < order; i++) {
                double arg = nodes[i] - espOrder / 2.0 + (dof - idof - 1);
                arg /= espOrder / 2.0;
                fn_v[idof * order + i] = pfun_w.eval(arg);
            }
        }

        // Chebyshev interpolation for order estimation
        std::vector<double> cheb_coeff;
        chebInterp1D(order, fn_v, cheb_coeff);

        int est_order = -1;
        for (int idof = 0; idof < dof; idof++) {
            double max_c = 0.0;
            for (int i = 0; i < order; i++)
                max_c = std::max(max_c, std::abs(cheb_coeff[idof * order + i]));
            for (int i = 0; i < order; i++) {
                if (std::abs(cheb_coeff[idof * order + i]) > tol_coeff_spread * max_c)
                    est_order = std::max(est_order, i + 1);
            }
        }
        est_order = std::min(std::max(est_order, 2), max_poly_order);

        // Build monomial coefficients at estimated order
        int nnodes = est_order;
        chebNodes1D(nnodes, nodes, 0, 1);
        fn_v.resize(dof * nnodes);
        for (int idof = 0; idof < dof; idof++) {
            for (int i = 0; i < nnodes; i++) {
                double arg = nodes[i] - espOrder / 2.0 + (dof - idof - 1);
                arg /= espOrder / 2.0;
                fn_v[idof * nnodes + i] = pfun_w.eval(arg);
            }
        }

        std::vector<double> coeffs_tmp;
        monomialInterp1D(est_order, nnodes, fn_v, coeffs_tmp);

        // Store in dense poly-major layout [est_order * P], direct order (no storage reversal).
        // The fn_v building already uses (dof-idof-1) to reverse the stencil order
        // (matching GROMACS convention). An additional reversal here would cancel it.
        // scoeffs[ip] for grid point gridIndex+ip must evaluate
        // spread_window_ref(c, P, P-1-ip, dr) = psi((dr + P/2 - 1 - ip)/(P/2)).
        esp.spreadPolyOrder = est_order;
        esp.spreadCoeffs.resize(espOrder * est_order);
        for (int ip = 0; ip < espOrder; ++ip) {
            int idof = ip;
            for (int j = 0; j < est_order; ++j) {
                esp.spreadCoeffs[j * espOrder + ip] = (float)coeffs_tmp[idof * est_order + j];
            }
        }
    }

    // The gather force directly fits the derivative of the PSWF window,
    // following the CPU ESP spreadRealDerivativePoly() path.
    {
        int order = MAX_CHEB_ORDER;
        std::vector<double> nodes;
        chebNodes1D(order, nodes);
        const double dsDx = 2.0/espOrder;
        std::vector<double> fn_v(espOrder*order);
        for (int idof = 0; idof < espOrder; idof++) {
            for (int i = 0; i < order; i++) {
                double arg = nodes[i] - espOrder/2.0 + (espOrder-idof-1);
                arg /= espOrder/2.0;
                fn_v[idof*order+i] = dsDx*pfun_w.evalDerivative(arg);
            }
        }

        std::vector<double> cheb_coeff;
        chebInterp1D(order, fn_v, cheb_coeff);

        int est_order = -1;
        for (int idof = 0; idof < espOrder; idof++) {
            double max_c = 0.0;
            for (int i = 0; i < order; i++)
                max_c = std::max(max_c, std::abs(cheb_coeff[idof*order+i]));
            for (int i = 0; i < order; i++) {
                if (std::abs(cheb_coeff[idof*order+i]) > tol_coeff_spread_der*max_c)
                    est_order = std::max(est_order, i+1);
            }
        }
        est_order = std::min(std::max(est_order, 2), max_poly_order);

        int nnodes = est_order;
        chebNodes1D(nnodes, nodes, 0, 1);
        fn_v.resize(espOrder*nnodes);
        for (int idof = 0; idof < espOrder; idof++) {
            for (int i = 0; i < nnodes; i++) {
                double arg = nodes[i] - espOrder/2.0 + (espOrder-idof-1);
                arg /= espOrder/2.0;
                fn_v[idof*nnodes+i] = dsDx*pfun_w.evalDerivative(arg);
            }
        }

        std::vector<double> coeffs_tmp;
        monomialInterp1D(est_order, nnodes, fn_v, coeffs_tmp);
        esp.spreadDerPolyOrder = est_order;
        esp.spreadDerCoeffs.resize(espOrder*esp.spreadDerPolyOrder);
        for (int ip = 0; ip < espOrder; ++ip)
            for (int j = 0; j < esp.spreadDerPolyOrder; ++j)
                esp.spreadDerCoeffs[j*espOrder+ip] = (float) coeffs_tmp[ip*esp.spreadDerPolyOrder+j];
    }

    // --- Fourier splitting coefficients use splitC (splitting PSWF) ---
    // splitFourier(x) = splitLambda * psi_s(x) / splitIntegral for x in [0, 1]
    {
        int order = MAX_CHEB_ORDER;
        std::vector<double> nodes;
        chebNodes1D(order, nodes);

        std::vector<double> fn_v(order);
        for (int i = 0; i < order; i++) {
            fn_v[i] = esp.splitLambda * pfun_s.eval(nodes[i]) / esp.splitIntegral;
        }

        std::vector<double> cheb_coeff;
        chebInterp1D(order, fn_v, cheb_coeff);

        int est_order = -1;
        double max_c = 0.0;
        for (int i = 0; i < order; i++)
            max_c = std::max(max_c, std::abs(cheb_coeff[i]));
        for (int i = 0; i < order; i++) {
            if (std::abs(cheb_coeff[i]) > tol_coeff_fourier * max_c)
                est_order = std::max(est_order, i + 1);
        }
        est_order = std::min(std::max(est_order, 2), max_poly_order);

        int nnodes = est_order;
        chebNodes1D(nnodes, nodes, 0, 1);
        fn_v.resize(nnodes);
        for (int i = 0; i < nnodes; i++) {
            fn_v[i] = esp.splitLambda * pfun_s.eval(nodes[i]) / esp.splitIntegral;
        }

        std::vector<double> coeffs_tmp;
        monomialInterp1D(est_order, nnodes, fn_v, coeffs_tmp);

        esp.splitFourierCoeffs.resize(est_order);
        for (int i = 0; i < est_order; ++i)
            esp.splitFourierCoeffs[i] = coeffs_tmp[i];
    }

    // --- Direct-space smooth long-range scale uses splitC (splitting PSWF) ---
    {
        const int maxDirectOrder = 40;
        fitLongRangeSplitMonomials(tol, maxDirectOrder, pfun_s, esp.splitIntegral,
                                   esp.longRangeEnergyCoeffs, esp.longRangeForceCoeffs,
                                   directEnergyTolFactor, directForceTolFactor);
    }

    return esp;
}

// Tolerance-based with explicit P override.
static inline EspCoefficients buildEspCoefficients(double tol, int P,
                                                             int max_poly_order = 16,
                                                             double directEnergyTolFactor = 0.001,
                                                             double directForceTolFactor = 0.1,
                                                             double spreadTolFactor = 1.0,
                                                             double spreadDerTolFactor = 1.0) {
    double splitC = getProlateC(tol);
    double windowC = getProlateC(0.5*tol);
    return buildEspCoefficients(splitC, windowC, P, tol, max_poly_order, directEnergyTolFactor, directForceTolFactor,
                                spreadTolFactor, spreadDerTolFactor);
}

// ============================================================================
// PSWF moduli for Fourier deconvolution
// ============================================================================

// Inverse PSWF moduli for Fourier deconvolution (multiply instead of divide).
// Modulus = (h * lambda * psi(zarg*k))^2 where h = P/2.
// NOTE: This uses the continuous FT value (single alias n=0 only).
// For typical ESP parameters (P>=16), scale*gridSize = pi*P/c > 2,
// meaning no aliases fall within PSWF support [-1,1], so the continuous
// approximation is exact. For small P or large c where scale*gridSize < 2,
// an aliased sum would improve accuracy.
static inline std::vector<float> computePswfInvModuli(int n, int order,
                                                          const Pswf0& pfun,
                                                          double c, double lambda) {
    std::vector<float> inv_mod(n);
    double scale = M_PI * order / ((double)n * c);
    double h = order / 2.0;  // P/2 factor: DFT of spread window = h * lambda * psi(zarg*k)
    int maxk = (n + 1) / 2;

    // Threshold: if f² is below this, treat as zero to avoid huge inv_mod values.
    // The k=0 value (h * lambda * psi(0))² is the largest; use it as reference.
    double f0 = h * lambda * pfun.eval(0.0);
    double f0_sq = f0 * f0;
    double threshold = f0_sq * 1e-20;  // 10 orders of magnitude below peak

    // k = 0
    inv_mod[0] = (f0_sq > threshold) ? (float)(1.0 / f0_sq) : 0.0f;

    // Positive k
    for (int k = 1; k < maxk; ++k) {
        double arg = scale * k;
        if (arg > 1.0) {
            inv_mod[k] = 0.0f;  // Out of support: multiply by 0
        } else {
            double f = h * lambda * pfun.eval(arg);
            double f2 = f * f;
            inv_mod[k] = (f2 > threshold) ? (float)(1.0 / f2) : 0.0f;
        }
    }

    // Negative k (mapped to n+k for k < 0)
    for (int k = -maxk; k < 0; ++k) {
        double arg = -scale * k;
        if (arg > 1.0) {
            inv_mod[n + k] = 0.0f;
        } else {
            double f = h * lambda * pfun.eval(arg);
            double f2 = f * f;
            inv_mod[n + k] = (f2 > threshold) ? (float)(1.0 / f2) : 0.0f;
        }
    }

    return inv_mod;
}
} // namespace pswf

#endif // OPENMM_ESP_PSWF_H_
