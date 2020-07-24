import numpy as np

class WaterfillingCompression:

    def __init__(self, num_coeffs, coeffs_to_keep, algo):
        self.num_coeffs = num_coeffs
        self.coeffs_to_keep = coeffs_to_keep
        self.algo = algo

    def compute_distance_bounds(self, data):
        (N, dim) = np.shape(data)
        # data in fourier basis (without storing complex conjugates)
        X_f = np.zeros((N, np.ceil(dim/2 + 1)))
        # check if the length is even
        is_even = (dim % 2 == 0)

        for i in range(N):
            x = data[i]
            fx = np.fft.fft(x)
            # include DC
            if is_even:
                fx_short = fx[:(dim/2 + 1)]
                fx_short[0] = fx_short/np.sqrt(2) # divide DC by sqrt(2)
            else:
                fx_short = fx[:np.ceil(dim/2 + 1)]
                fx_short = np.concatenate((fx_short[1:], np.array(fx_short[0]))) # bring DC to the end
            X_f[i] = fx_short

        # initialize lower and upper bound distance matrices
        dm_ub = np.zeros((N, N))
        dm_lb = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                ub, lb = self.dist_cc(X_f[i], X_f[j], self.num_coeffs, self.coeffs_to_keep)
                # ignore imaginary part
                dm_ub[i, j] = np.real(ub)
                dm_lb[i, j] = np.real(lb)
        return dm_ub, dm_lb

    def dist_cc(self, x1, x2, num_coeffs, coeffs_to_keep):
        N = len(x1)
        # For avoiding to store complex conjugates
        x1[-1] = x1[-1] / np.sqrt(2)
        x2[-1] = x2[-1] / np.sqrt(2)
        power_x1 = np.absolute(x1)
        power_x2 = np.absolute(x2)

        # wich coefficients to keep
        if coeffs_to_keep == "first":
            coeffs_x1 = list(range(num_coeffs))
            other_coeffs_x1 = list(range(num_coeffs, N))
            coeffs_x2 = list(range(num_coeffs))
            other_coeffs_x2 = list(range(num_coeffs, N))
        else: # coeffs_to_keep =  "best" or "optimal"
            sorted_indices_x1 = np.argsort(power_x1)[::-1]
            coeffs_x1 = sorted_indices_x1[:num_coeffs]
            other_coeffs_x1 = sorted_indices_x1[num_coeffs:]
            sorted_indices_x2 = np.argsort(power_x2)[::-1]
            coeffs_x2 = sorted_indices_x2[:num_coeffs]
            other_coeffs_x2 = sorted_indices_x2[num_coeffs:]

        # compression error
        e_x1 = np.linalg.norm(x1[other_coeffs_x1]) ** 2
        e_x2 = np.linalg.norm(x2[other_coeffs_x2]) ** 2

        P0 = np.intersect1d(coeffs_x1, coeffs_x2)  # coefficients where x1_i and x2_i are known
        P1 = np.intersect1d(other_coeffs_x1, coeffs_x2)  # coefficients where x1_i is unknown and x2_i is known
        P2 = np.intersect1d(coeffs_x1, other_coeffs_x2)  # coefficients where x1_i is known and x2_i is unknown
        P3 = np.intersect1d(other_coeffs_x1, other_coeffs_x2)  # coefficients where x1_i and x2_i are unknown

        if coeffs_to_keep == "first":
            # all coefficients are the same
            lb = np.sqrt(2) * np.linalg.norm(x1[P0] - x2[P0])  # sqrt(2) needed for storing only one conjugate pair
            ub = lb
        elif coeffs_to_keep == "best":
            lb = np.sqrt(2) * np.sqrt(np.linalg.norm(x1[P0] - x2[P0]) ** 2
                                      + np.linalg.norm(x1[P2] ** 2)
                                      + np.linalg.norm(x2[P1]) ** 2)
            ub = lb
        else: # coeffs_to_keep = "optimal"
            if len(P0) > 0:
                dist = 2 * (np.linalg.norm(x1)**2 + np.linalg.norm(x2)**2 - 2 * np.real(np.inner(x1[P0], x2[P0])))
            else:
                dist = 2 * (np.linalg.norm(x1)**2 + np.linalg.norm(x2)**2)  # 2 * needed for fourier
            x1_max = np.min(power_x1[coeffs_x1])
            x2_max = np.min(power_x2[coeffs_x2])
            # Double waterfilling
            res = self.double_waterfill(power_x1[P2], power_x2[P1], e_x1, e_x2, x1_max, x2_max, P1, P2, P3)
            # *2 for expansion in quadratic
            lb = np.sqrt(dist - 4 * res)
            ub = np.sqrt(dist + 4 * res)
        return ub, lb

    def double_waterfill(self, a, b, e_a, e_b, a_max, b_max, P1, P2, P3):
        p_x1_minus = np.union1d(P1, P3)
        p_x2_minus = np.union1d(P2, P3)

        def waterfill(a, e_a, a_max):
            N = len(a)
            if N == 0:
                res = 0
                x = np.array([])
                E = e_a
                return res, x, E
            a = np.absolute(a)
            if N * a_max**2 <= e_a:
                x = a_max * np.ones(N)
                E = e_a - N * a_max**2
                res = np.inner(a, x)
            else:
                R = e_a
                c = np.array(range(N))
                x = np.zeros(N)
                while (R > 0) & len(c) > 0:
                    x[c] = a[c] / (np.linalg.norm(a[c]) / np.sqrt(R))
                    i = np.where(x[c] >= a_max)[0]
                    if len(i) == 0:
                        break
                    x[c[i]] = a_max
                    c = np.delete(c, i)
                    R = e_a - (N - len(c)) * (a_max ** 2)

                res = np.inner(a, x)
                E = 0
            return res, x, E

        def gamma_opt():
            return 0

        if len(np.intersect1d(p_x1_minus, p_x2_minus)) == 0:
            if len(p_x1_minus) == 0: # all coefficients of a known, P1=P3=0
                return waterfill(a, e_b, b)
            if len(p_x2_minus) == 0: # all coefficients of b known, P2=P3=0
                return waterfill(b, e_a, a)
            if len(P3) == 0:
                res1 = waterfill(b, e_a, a_max)
                res2 = waterfill(a, e_b, b_max)
                return res1 + res2
        if (len(P1) == 0) & (len(P2) == 0):
            return np.sqrt(e_a) * np.sqrt(e_b)
        if (e_a <= len(P1) * a_max**2) & (e_b <= len(P2) * b_max**2):
            res1 = waterfill(b, e_a, a_max)
            res2 = waterfill(a, e_b, b_max)
            return res1 + res2
        else:
            gamma = gamma_opt()
            if gamma == -1:
                return 0
            e_a_prime = e_a - np.sum(np.minimum((b ** 2) * gamma, (a_max ** 2) * np.ones(len(b))))
            e_b_prime = e_b - np.sum(np.minimum((a ** 2) / gamma, (b_max ** 2) * np.ones(len(a))))
            res1 = waterfill(b, e_a-e_a_prime, a_max)
            res2 = waterfill(a, e_b-e_b_prime, b_max)
            res3 = np.sqrt(e_a_prime) * np.sqrt(e_b_prime)
            return res1 + res2 + res3

