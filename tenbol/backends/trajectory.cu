// One CUDA thread owns a mode for the whole integration. State is private to
// the thread. Only the coefficient tables and requested outputs cross the bus.

enum TableColumn { TAU, KAPPA, DKAPPA_DLOG_A, CS2, NU_PRESSURE, NU_DENSITY };
enum BackgroundColumn { COLD, BARYON, PHOTON, RADIATION, VACUUM, MASS };
constexpr double NO_EVENT = 1e300;

struct Grid {
    const double* rows;
    double origin, spacing;
    int count;

    __device__ const double* row(int i) const { return rows + i * (NU_DENSITY + NS); }
    __device__ int interval(double x) const {
        return max(0, min(count - 2, int((x - origin) / spacing)));
    }
    __device__ double crossing(int i, int field, double target) const {
        double left = row(i)[field], right = row(i + 1)[field];
        return origin + (i + (target - left) / (right - left)) * spacing;
    }
    __device__ double at_tau(double target) const {
        if (target <= row(0)[TAU]) return -NO_EVENT;
        if (target > row(count - 1)[TAU]) return NO_EVENT;
        int lo = 0, hi = count - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (row(mid)[TAU] < target) lo = mid;
            else hi = mid;
        }
        return crossing(lo, TAU, target);
    }
    __device__ double next_opacity_crossing(double target, int& i) const {
        for (; i < count - 1; ++i) {
            if ((row(i)[KAPPA] > target) != (row(i + 1)[KAPPA] > target)) {
                return crossing(i++, KAPPA, target);
            }
        }
        return NO_EVENT;
    }
};

struct Coefficients {
    double tau, kappa, dopacity, cs2, h, gpres, cold, baryon, photon, radiation;
    double nu_density[NS > 0 ? NS : 1];
    double mass[NS > 0 ? NS : 1];
};

__device__ Coefficients coefficients(Grid grid, const double* bg, double x) {
    const int i = grid.interval(x);
    const double t = (x - grid.origin) / grid.spacing - i;
    const double* lo = grid.row(i);
    const double* hi = grid.row(i + 1);
    const double a = exp(x), inv_a = 1.0 / a, inv_a2 = inv_a * inv_a;
    Coefficients c;
    c.tau = lo[TAU] * (1.0 - t) + hi[TAU] * t;
    c.kappa = fmin(lo[KAPPA] * (1.0 - t) + hi[KAPPA] * t, 1e30);
    c.cs2 = lo[CS2] * (1.0 - t) + hi[CS2] * t;
    c.cold = bg[COLD] * inv_a;
    c.baryon = bg[BARYON] * inv_a;
    c.photon = bg[PHOTON] * inv_a2;
    c.radiation = bg[RADIATION] * inv_a2;
    const double vacuum = bg[VACUUM] * a * a;
    double rho = c.cold + c.baryon + c.photon + c.radiation + vacuum;
    #pragma unroll
    for (int s = 0; s < NS; ++s) {
        c.nu_density[s] = (lo[NU_DENSITY + s] * (1.0 - t) + hi[NU_DENSITY + s] * t) * inv_a2;
        c.mass[s] = a * bg[MASS + s];
        rho += c.nu_density[s];
    }
    c.h = sqrt(rho / 3.0);
    c.gpres = (c.photon + c.radiation) / 3.0 - vacuum
        + (lo[NU_PRESSURE] * (1.0 - t) + hi[NU_PRESSURE] * t) * inv_a2;
    c.dopacity = (lo[DKAPPA_DLOG_A] * (1.0 - t) + hi[DKAPPA_DLOG_A] * t) * c.h;
    return c;
}

struct Regime { bool tc, rsa_g, rsa_r, freeze_nu, shorten_nu; };

__device__ double down(double k, int l) { return k * l / (2.0 * l + 1.0); }
__device__ double up(double k, int l) { return k * (l + 1.0) / (2.0 * l + 1.0); }
__device__ int fg(int l) { return l == 2 ? PIG : FG + l - 3; }
__device__ int pol(int l) { return POL + l - 2; }
__device__ int fr(int l) { return l == 2 ? PIR : FR + l - 3; }
__device__ int nu(int s, int q) { return FNU + (s * NQ + q) * (LNU + 1); }

__device__ double energy(double mass, double q) {
    const double u = mass / q;
    return sqrt(1.0 + u * u);
}

// The device implementation shares the CPU solver's state layout and physics
// settings. The regime masks are constant over a step, with one-sided values
// at a switch. Return the tight-coupling quadrupole for the exit state.
__device__ double rhs(
    const double* y, double* f, const Coefficients& c, double k, Regime regime,
    const double* momenta, const double* weights, int short_l, int freeze_l
) {
    double density_nu = 0.0, flux_nu = 0.0;
    for (int s = 0; s < NS; ++s) {
        double denominator = 0.0, density = 0.0, flux = 0.0;
        for (int q = 0; q < NQ; ++q) {
            const double w = weights[q], we = w * energy(c.mass[s], momenta[q]);
            const int i = nu(s, q);
            denominator += we;
            density += we * y[i];
            flux += w * y[i + 1];
        }
        density_nu += c.nu_density[s] * (density / denominator);
        flux_nu += c.nu_density[s] * (flux / denominator);
    }
    const double dm = c.baryon * y[DB] + c.cold * y[DC] + density_nu;
    const double qm = c.baryon * y[VB] + flux_nu;
    const double inv_h = 1.0 / c.h, inv_tau = 1.0 / fmax(c.tau, 1e-30);
    const double zm = (0.5 * dm / k + y[ETAK]) * inv_h;
    const double dzm = -c.h * zm - 0.5 * dm / k;
    const double dg = regime.rsa_g ? -4.0 / k * (dzm + c.kappa * (y[VB] + zm)) : y[DG];
    const double qg = regime.rsa_g ? -4.0 / 3.0 * zm : y[QG];
    const double dr = regime.rsa_r ? -4.0 / k * dzm : y[DR];
    const double qr = regime.rsa_r ? -4.0 / 3.0 * zm : y[QR];
    const double dgrho = dm + c.photon * dg + c.radiation * dr;
    const double dgq = qm + c.photon * qg + c.radiation * qr;
    const double z = (0.5 * dgrho / k + y[ETAK]) * inv_h;
    const double sigma = z + 1.5 * dgq / (k * k);
    const double metric = (8.0 / 15.0) * k * sigma;
    const double dbdot = -k * (z + y[VB]);
    const double dgdot = -k * (4.0 / 3.0 * z + qg);
    const double photbar = c.photon / c.baryon, pb43 = (4.0 / 3.0) * photbar;
    const double vb = y[VB], pig = y[PIG], pir = y[PIR];
    const double pig_exit = 32.0 / 45.0 / fmax(c.kappa, 1e-300) * k * (sigma + vb);
    double pig_tc = pig_exit;
    double vbdot, qgdot;
    if (regime.tc) {
        const double adotdota = (c.h * c.h - c.gpres) / 2.0;
        const double slip = -(2.0 * c.h / (1.0 + pb43) + c.dopacity / c.kappa)
            * (vb - 0.75 * qg)
            + (-adotdota * vb - k / 2.0 * c.h * dg + k * (c.cs2 * dbdot - dgdot / 4.0))
            / (c.kappa * (1.0 + pb43));
        const double sigmadot = -2.0 * c.h * sigma - (c.photon * pig_tc + c.radiation * pir) / k
            + y[ETAK];
        const double qgdot_tc = k * (dg / 4.0 - pig_tc / 2.0) + c.kappa * slip;
        const double kap2 = c.kappa * c.kappa;
        pig_tc = 32.0 / 45.0 / c.kappa * k * (sigma + 0.75 * qg)
            * (1.0 + c.dopacity * (11.0 / 6.0) / kap2)
            - 32.0 / 45.0 / kap2 * k * (sigmadot + 0.75 * qgdot_tc) * (11.0 / 6.0);
        vbdot = (-c.h * vb + c.cs2 * k * y[DB] + k / 4.0 * pb43 * (dg - 2.0 * pig_tc))
            / (1.0 + pb43) + pb43 / (1.0 + pb43) * slip;
        qgdot = (4.0 / 3.0) * (-vbdot - c.h * vb + c.cs2 * y[DB] * k) / pb43
            + k / 3.0 * dg - 2.0 / 3.0 * k * pig_tc;
    } else {
        vbdot = -c.h * vb + k * c.cs2 * y[DB] - photbar * c.kappa * (4.0 / 3.0 * vb - qg);
        qgdot = k / 3.0 * dg - 2.0 / 3.0 * k * pig + 4.0 / 3.0 * c.kappa * (vb - 0.75 * qg);
    }
    f[DC] = -k * z * inv_h;
    f[DB] = dbdot * inv_h;
    f[VB] = vbdot * inv_h;
    f[ETAK] = 0.5 * dgq * inv_h;
    f[DG] = regime.rsa_g ? 0.0 : dgdot * inv_h;
    f[QG] = regime.rsa_g ? 0.0 : qgdot * inv_h;

    if (regime.tc || regime.rsa_g) {
        f[PIG] = 0.0;
        for (int i = FG; i < FR; ++i) f[i] = 0.0;
    } else {
        const double polter = pig / 10.0 + 0.6 * y[pol(2)];
        f[PIG] = (down(k, 2) * qg - up(k, 2) * y[fg(3)]
            - c.kappa * (pig - polter) + metric) * inv_h;
        for (int l = 3; l < LG; ++l) {
            f[fg(l)] = (down(k, l) * y[fg(l - 1)] - up(k, l) * y[fg(l + 1)]
                - c.kappa * y[fg(l)]) * inv_h;
        }
        f[fg(LG)] = (k * y[fg(LG - 1)] - (LG + 1) * y[fg(LG)] * inv_tau
            - c.kappa * y[fg(LG)]) * inv_h;
        f[pol(2)] = (-c.kappa * (y[pol(2)] - polter) - k / 3.0 * y[pol(3)]) * inv_h;
        for (int l = 3; l < LP; ++l) {
            const double polfack = (l + 3.0) * (l - 1.0) / (l + 1.0) * k / (2.0 * l + 1.0);
            f[pol(l)] = (-c.kappa * y[pol(l)] + down(k, l) * y[pol(l - 1)]
                - polfack * y[pol(l + 1)]) * inv_h;
        }
        f[pol(LP)] = (-c.kappa * y[pol(LP)] + k * (LP / double(LP - 2)) * y[pol(LP - 1)]
            - (LP + 3) * y[pol(LP)] * inv_tau) * inv_h;
    }

    if (regime.rsa_r) {
        f[DR] = f[QR] = f[PIR] = 0.0;
        for (int i = FR; i < FNU; ++i) f[i] = 0.0;
    } else {
        f[DR] = -k * (4.0 / 3.0 * z + qr) * inv_h;
        f[QR] = (down(k, 1) * dr - up(k, 1) * pir) * inv_h;
        f[PIR] = (down(k, 2) * qr - up(k, 2) * y[fr(3)] + metric) * inv_h;
        for (int l = 3; l < LR; ++l) {
            f[fr(l)] = (down(k, l) * y[fr(l - 1)] - up(k, l) * y[fr(l + 1)]) * inv_h;
        }
        f[fr(LR)] = (down(k, LR) * y[fr(LR - 1)] - (LR + 1) * y[fr(LR)] * inv_tau) * inv_h;
    }

    const int top = regime.shorten_nu ? short_l : LNU;
    const int active = regime.freeze_nu ? min(top + 1, freeze_l) : top + 1;
    if (active == 0) {
        for (int i = FNU; i < DIM; ++i) f[i] = 0.0;
    } else {
        for (int s = 0; s < NS; ++s) {
            for (int q = 0; q < NQ; ++q) {
                const int i = nu(s, q);
                const double v = 1.0 / energy(c.mass[s], momenta[q]);
                f[i] = -k * (4.0 / 3.0 * z + v * y[i + 1]) * inv_h;
                for (int l = 1; l < active; ++l) {
                    if (l == top) {
                        f[i + l] = (k * v * y[i + l - 1] - (l + 1) * y[i + l] * inv_tau) * inv_h;
                    } else {
                        f[i + l] = (v * (down(k, l) * y[i + l - 1] - up(k, l) * y[i + l + 1])
                            + (l == 2 ? metric : 0.0)) * inv_h;
                    }
                }
                for (int l = active; l <= LNU; ++l) f[i + l] = 0.0;
            }
        }
    }
    return pig_exit;
}

__device__ double interpolate(double y, double end, double f, double f_end, double dx, double t) {
    const double t2 = t * t, t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * y + (t3 - 2.0 * t2 + t) * dx * f
        + (-2.0 * t3 + 3.0 * t2) * end + (t3 - t2) * dx * f_end;
}

extern "C" __global__ void integrate(
    const double* initial, const double* wave, const int* sample, const double* background,
    const double* tables, const double* switches, const double* momenta, const double* weights,
    const double* output_x, double* result, int* steps, double* last_x,
    int n, int nout, int grid_count, double x_origin, double grid_spacing, double x_start,
    int max_steps, int short_l, int freeze_l, double tc_epsilon, double rsa_ktau,
    double rsa_ktau_r, double rsa_ktau_nu, double kmin_g, double kmin_r,
    double dx_min, double dx_max, double acoustic, double nu_cfl, double horizon, double stiff
) {
    const int trajectory = blockIdx.x * blockDim.x + threadIdx.x;
    if (trajectory >= n) return;
    const int b = sample[trajectory];
    const double* bg = background + b * (MASS + NS);
    Grid grid{tables + b * grid_count * (NU_DENSITY + NS), x_origin, grid_spacing, grid_count};
    const double k = wave[trajectory], tc_threshold = k / tc_epsilon;
    const double tau_g = switches[trajectory], tau_nu = switches[n + trajectory];
    const double x_rsa_g = k > kmin_g ? grid.at_tau(fmax(rsa_ktau / k, tau_g)) : NO_EVENT;
    const double x_rsa_r = k > kmin_r ? grid.at_tau(fmax(rsa_ktau_r / k, tau_nu)) : NO_EVENT;
    const double x_freeze = k > kmin_r ? grid.at_tau(fmax(rsa_ktau_nu / k, tau_nu)) : NO_EVENT;
    const double x_short = k > kmin_r ? grid.at_tau(fmax(20.0, LNU - 4.0) / k) : NO_EVENT;

    double state[DIM], work[DIM], dy[DIM], increment[DIM];
    double* y = state;
    double* stage = work;
    for (int i = 0; i < DIM; ++i) y[i] = initial[i * n + trajectory];
    double x = x_start;
    Coefficients c = coefficients(grid, bg, x);
    bool tc = c.kappa > tc_threshold;
    int tc_interval = grid.interval(x);
    double x_tc = grid.next_opacity_crossing(tc_threshold, tc_interval);
    if (x_tc <= x) x_tc = grid.next_opacity_crossing(tc_threshold, tc_interval);
    int next_output = 0;

    for (int step = 0; step < max_steps && next_output < nout; ++step) {
        const Regime regime{tc, x >= x_rsa_g, x >= x_rsa_r, x >= x_freeze, x >= x_short};
        const double k_over_h = k / c.h;
        double dx = dx_max;
        if (!tc && !regime.rsa_g) dx = fmin(dx, acoustic / k_over_h);
        if (!regime.rsa_r || (NS > 0 && (!regime.freeze_nu || freeze_l > 0))) {
            dx = fmin(dx, nu_cfl / k_over_h);
        }
        if (k * c.tau < 1.0) dx = fmin(dx, horizon);
        if (!tc) {
            const double scattering = c.kappa / c.h;
            const double drag = scattering * (1.0 + (4.0 / 3.0) * c.photon / c.baryon);
            dx = fmin(dx, stiff / fmax(scattering, 1e-30));
            // The photon-baryon slip decays faster than opacity alone when
            // tight coupling ends in a radiation-dominated mode. Keep the
            // dissipative mode inside RK4's negative-real stability interval.
            dx = fmin(dx, 1.8 / fmax(drag, 1e-30));
        }
        double end = fmin(x + fmax(dx, dx_min), output_x[nout - 1]);
        end = fmin(end, x_tc);
        if (!regime.rsa_g) end = fmin(end, x_rsa_g);
        if (!regime.rsa_r) end = fmin(end, x_rsa_r);
        if (!regime.freeze_nu) end = fmin(end, x_freeze);
        if (!regime.shorten_nu) end = fmin(end, x_short);
        const double h = end - x, half = h * 0.5;
        const Coefficients mid = coefficients(grid, bg, x + half);
        const Coefficients finish = coefficients(grid, bg, end);

        rhs(y, dy, c, k, regime, momenta, weights, short_l, freeze_l);
        const double f_dc = dy[DC], f_db = dy[DB];
        double f_nu[NS * NQ > 0 ? NS * NQ : 1];
        if (output_x[next_output] <= end) {
            for (int s = 0; s < NS; ++s) {
                for (int q = 0; q < NQ; ++q) f_nu[s * NQ + q] = dy[nu(s, q)];
            }
        }
        for (int i = 0; i < DIM; ++i) {
            increment[i] = dy[i];
            stage[i] = y[i] + half * dy[i];
        }
        rhs(stage, dy, mid, k, regime, momenta, weights, short_l, freeze_l);
        for (int i = 0; i < DIM; ++i) {
            increment[i] += 2.0 * dy[i];
            stage[i] = y[i] + half * dy[i];
        }
        rhs(stage, dy, mid, k, regime, momenta, weights, short_l, freeze_l);
        for (int i = 0; i < DIM; ++i) {
            increment[i] += 2.0 * dy[i];
            stage[i] = y[i] + h * dy[i];
        }
        rhs(stage, dy, finish, k, regime, momenta, weights, short_l, freeze_l);
        bool finite = true;
        for (int i = 0; i < DIM; ++i) {
            stage[i] = y[i] + h / 6.0 * (increment[i] + dy[i]);
            finite &= isfinite(stage[i]);
        }
        if (!finite) {
            steps[trajectory] = -step - 1;
            last_x[trajectory] = end;
            return;
        }
        while (next_output < nout && output_x[next_output] <= end) {
            const double at = output_x[next_output];
            const double t = h > 0.0 ? fmax(0.0, (at - x) / h) : 0.0;
            const double dc = interpolate(y[DC], stage[DC], f_dc, dy[DC], h, t);
            const double db = interpolate(y[DB], stage[DB], f_db, dy[DB], h, t);
            const Coefficients out = coefficients(grid, bg, at);
            double contrast = out.cold * dc + out.baryon * db;
            double total = out.cold + out.baryon;
            for (int s = 0; s < NS; ++s) {
                double density = 0.0, denominator = 0.0;
                for (int q = 0; q < NQ; ++q) {
                    const int i = nu(s, q);
                    const double we = weights[q] * energy(out.mass[s], momenta[q]);
                    density += we * interpolate(y[i], stage[i], f_nu[s * NQ + q], dy[i], h, t);
                    denominator += we;
                }
                contrast += out.nu_density[s] * density / denominator;
                total += out.nu_density[s];
            }
            const int index = (next_output * n + trajectory) * 3;
            result[index] = dc;
            result[index + 1] = db;
            result[index + 2] = contrast / total;
            ++next_output;
        }
        if (end == x_tc) {
            if (tc) {
                const double pig_tc = rhs(stage, dy, finish, k, regime,
                                          momenta, weights, short_l, freeze_l);
                stage[PIG] = pig_tc;
                stage[pol(2)] = pig_tc / 4.0;
                stage[pol(3)] = stage[fg(3)] / 4.0;
            }
            tc = !tc;
            x_tc = grid.next_opacity_crossing(tc_threshold, tc_interval);
        }
        double* previous = y;
        y = stage;
        stage = previous;
        c = finish;
        x = end;
        if (next_output == nout) {
            steps[trajectory] = step + 1;
            last_x[trajectory] = x;
            return;
        }
    }
    steps[trajectory] = 0;
    last_x[trajectory] = x;
}
