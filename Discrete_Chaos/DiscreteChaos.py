""" ...
"""

import itertools
import matplotlib.pyplot as plt
import time
import numpy as np


class DiscreteChaosSuite:
    """...
    """

    def __init__(self, iteration):
        self.iteration = iteration

    def next_point(self, point, pams):
        return self.iteration(point, pams)

    def sequence_gen_finite(self, point, pams, num_point=2000):
        for p in range(num_point):
            if np.any(np.isnan(point)):
                raise ValueError("Encountered NaN")
            yield point
            point = self.iteration(point, pams)

    def sequence_gen_infinite(self, point, pams):
        while True:
            if np.any(np.isnan(point)):
                raise ValueError("Encountered NaN")
            yield point
            point = self.iteration(point, pams)

    @staticmethod
    def _find_iterable_(pams):
        """
        Takes a tuple as an input and returns the first iterator this list contains
        :param pams:
        :return:
        """
        loc, iter = None, None
        for num, item in enumerate(pams):
            if hasattr(item, '__iter__'):
                loc, iter = num, item
                break
        if iter is None:
            raise TypeError('No iterator is given')
        return iter, loc

    def _iterator_parameter_gen_(self, pams):
        iter, loc = DiscreteChaosSuite._find_iterable_(pams)
        for p in iter:
            yield tuple(p if i == loc else pams[i] for i in range(len(pams)))

    def bifurcation_dict(self, start_point, parameters, num_points=2000, points_to_skip=100):
        """
        Returns a dictionary
        :param init_point: tuple, startint point for the map
        :param parameters: tuple, parameters for the map, should contain at leas one iterable
        :param kwargs: parameters for the scatter function
        :return:
        """
        t = time.time()
        if points_to_skip >= num_points:
            raise ValueError('The number of total points generated has to be greater than the number of points skipped')
        d = dict()
        _, pam_loc = self._find_iterable_(parameters)
        for pam in self._iterator_parameter_gen_(parameters):
            try:
                point_gen = self.sequence_gen_finite(start_point, pam, num_points)
                for k in range(points_to_skip):
                    next(point_gen)
                d[pam[pam_loc]] = point_gen
            except Exception as exc:
                print(f"Problem here {exc}")
                continue
        return d

    def bifurcation_diagram(self, start_point, pams, which_var=0, num_points=2000, points_to_skip=100, fig=None, **kwargs):

        # Find the position of the iterable in the pams tuple
        _, loc = self._find_iterable_(pams)

        # Create the bifurcation dictionary, where the keys are the parameter tuples
        d = self.bifurcation_dict(start_point, pams, num_points, points_to_skip)

        if fig is None:
            fig = plt.figure()
        for k in d.keys():
            p = k[loc]
            # points = list(dict[k][which_var])
            points = [f[which_var] for f in d[k]]
            ell = len(points)
            plt.scatter(ell * [p], points, **kwargs)
        return fig

    def return_map(self, init_point, parameter, num_points=2000, which_var=0, fig=None, **kwargs):
        seq_gen = self.sequence_gen_finite(init_point, parameter, num_points)
        points1 = [point[which_var] for point in seq_gen]
        points2 = points1[1:]
        points1 = points1[:-1]
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)
        plt.scatter(points1, points2, **kwargs)

    @staticmethod
    def _cobweb_line_(x, y, z):
        x_locs = [x, y, y]
        y_locs = [y, y, z]
        return x_locs, y_locs

    def cobweb_diagram(self, init_point, parameter, num_points=50, which_var=0, fig=None, **kwargs):
        seq_gen = self.sequence_gen_finite(init_point, parameter, num_points)
        points = [point[which_var] for point in seq_gen]
        x_plot = []
        y_plot = []
        for k in range(len(points) - 2):
            x_n, y_n = DiscreteChaosSuite._cobweb_line_(*points[k: k+3])
            x_plot.extend(x_n)
            y_plot.extend(y_n)
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)
        plt.plot(x_plot, y_plot, **kwargs)
        plt.xlim(min(x_plot), max(x_plot))
        plt.ylim(min(y_plot), max(y_plot))
        plt.plot([min(x_plot), max(x_plot)], [min(x_plot), max(x_plot)], 'k')

        return fig

    @staticmethod
    def move_by_d(point, d):
        new_point = [point[k] + (-1)**k * d / np.sqrt(len(point)) for k in range(len(point))]
        new_point = tuple(new_point)
        return new_point

    @staticmethod
    def tuple_dist(t1, t2):
        l1, l2 = np.array(t1), np.array(t2)
        return np.linalg.norm(l1 - l2)

    def lyapunov_exponent(self, init_point, parameter, e=1e-08, num_points=2000, discard=200):
        # Find the position after discard number of iterations
        gen = self.sequence_gen_infinite(init_point, parameter)

        # Initialize the Lyapunov exponent
        lyapunov = 0
        # Initialize the base point inside the attractor
        base_point = next(itertools.islice(gen, discard, None))
        moved_point = DiscreteChaosSuite.move_by_d(base_point, e)
        base_point = self.iteration(base_point, parameter)
        moved_point = self.iteration(moved_point, parameter)
        for it in range(num_points):
            d = DiscreteChaosSuite.tuple_dist(base_point, moved_point)
            lyapunov += np.log(d) - np.log(e)
            # Normalize the point
            base_point = tuple([base_point[k] + d * (moved_point[k] - base_point[k])/e for k in range(len(base_point))])
            moved_point = DiscreteChaosSuite.move_by_d(base_point, e)
            base_point = self.iteration(base_point, parameter)
            moved_point = self.iteration(moved_point, parameter)
        # Return the mean
        return lyapunov/num_points

    def lyapunov_exponent_plot(self, init_point, parameters, e=1e-08, num_points=2000, discard=200, fig=None, **kwargs):
        _, loc = self._find_iterable_(parameters)
        pam_gen = self._iterator_parameter_gen_(parameters)

        x_plot, y_plot = [], []
        for p_set in pam_gen:
            l = self.lyapunov_exponent(init_point, p_set, e, num_points, discard)
            x_plot.append(p_set[loc])
            y_plot.append(l)

        if fig is None:
            plt.figure()
        else:
            plt.figure(fig.number)
        plt.plot(x_plot, y_plot, **kwargs)

    @staticmethod
    def move_specific_by_d(point, which, d):
        new_point = [point[k] + d if k == which else point[k] for k in range(len(point))]
        new_point = tuple(new_point)
        return new_point

    def lyapunov_exponent_choose_variable(self, init_point, parameter, e=1e-08, num_points=2000, discard=200, which_var=0):
        # Find the position after discard number of iterations
        gen = self.sequence_gen_infinite(init_point, parameter)

        # Initialize the Lyapunov exponent
        lyapunov = 0
        # Initialize the base point inside the attractor
        base_point = next(itertools.islice(gen, discard, None))
        moved_point = DiscreteChaosSuite.move_specific_by_d(base_point, which_var, e)
        base_point = self.iteration(base_point, parameter)
        moved_point = self.iteration(moved_point, parameter)

        for it in range(num_points):
            d = DiscreteChaosSuite.tuple_dist(base_point, moved_point)

            # d_var = abs(base_point[which_var] - moved_point[which_var])
            lyapunov += np.log(d) - np.log(e)
            # Normalize the point
            base_point = tuple([base_point[k] + d * (moved_point[k] - base_point[k])/e for k in range(len(base_point))])
            moved_point = DiscreteChaosSuite.move_specific_by_d(base_point, which_var, e)
            base_point = self.iteration(base_point, parameter)
            moved_point = self.iteration(moved_point, parameter)
        # Return the mean
        return lyapunov/num_points

    def lyapunov_exponent_plot_choose_variable(self, init_point, parameters, e=1e-08, num_points=2000, discard=200,\
                                               which_variable=0, fig=None, **kwargs):
        _, loc = self._find_iterable_(parameters)
        pam_gen = self._iterator_parameter_gen_(parameters)

        x_plot, y_plot = [], []
        for p_set in pam_gen:
            l = self.lyapunov_exponent_choose_variable(init_point, p_set, e, num_points, discard, which_variable)
            x_plot.append(p_set[loc])
            y_plot.append(l)

        if fig is None:
            plt.figure()
        else:
            plt.figure(fig.number)
        plt.plot(x_plot, y_plot, **kwargs)

    def double_lyapunov(self, init_point, parameters, h=1e-04, num_points=1000, discard=50, which_var=0, fig=None, **kwargs):
        # Find the first iterable in the parameters
        iter1, loc1 = self._find_iterable_(parameters)
        iter2, loc2 = self._find_iterable_(parameters[loc1+1:])
        loc2 = loc1 + 1

        colors = []
        x_axis = []
        y_axis = []
        for p1 in iter1:
            t = time.time()
            y_axis.extend(iter2)
            x_axis.extend(len(iter2)*[p1])
            for p2 in iter2:
                cur_parameters = tuple([p1 if k == loc1 else p2 if k == loc2 else parameters[k] for k in range(len(parameters))])
                l = self.lyapunov_exponents_approximate_qr(init_point, cur_parameters, num_points, discard, h=h)[which_var]
                if l > 0:
                    color = 'r'
                else:
                    color = 'k'
                colors.extend([color])
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig.number)
        kwargs['c'] = colors
        plt.scatter(x_axis, y_axis, **kwargs)

    def approximate_partial_derivative(self, point, pams, which_num=0, which_den=0, h=1e-4):
        p1 = DiscreteChaosSuite.move_specific_by_d(point, which_den, h)
        p2 = DiscreteChaosSuite.move_specific_by_d(point, which_den, -h)

        np1 = self.iteration(p1, pams)
        np2 = self.iteration(p2, pams)

        return (np1[which_num] - np2[which_num])/(2 * h)

    def approximate_jacobian(self, point, pam, h=1e-4):
        ind = range(len(point))
        j = np.array([self.approximate_partial_derivative(point, pam, i, j, h) for i in ind for j in ind])
        j = j.reshape((len(point), len(point)))
        return j

    def lyapunov_exponents_approximate_qr(self, point, pams, num_points=1000, disc=100, h=1e-04):

        gen = self.sequence_gen_finite(point, pams, num_points)
        gen = itertools.islice(gen, disc, None)

        diag_elements = []

        q_dag = np.identity(len(point))
        for p in gen:
            j = self.approximate_jacobian(p, pams, h)
            q = j @ q_dag
            q_dag, r1 = np.linalg.qr(q)
            diag_elements.append(np.diagonal(r1))

        les = np.zeros(diag_elements[0].shape)
        for d in diag_elements:
            les += np.log(np.abs(d))

        return les/num_points

    def lyapunov_exponent_dict(self, start_point, parameters, num_points=2000, points_to_skip=100):
        """
        Returns a dictionary
        :param init_point: tuple, startint point for the map
        :param parameters: tuple, parameters for the map, should contain at leas one iterable
        :param kwargs: parameters for the scatter function
        :return:
        """
        t = time.time()
        if points_to_skip >= num_points:
            raise ValueError('The number of total points generated has to be greater than the number of points skipped')
        d = {}
        for pam in self._iterator_parameter_gen_(parameters):
            try:
                le_now = self.lyapunov_exponents_approximate_qr(
                    start_point, pam, num_points=1000,
                    disc=100, h=1e-04
                    )
                _, pam_loc = self._find_iterable_(parameters)
                d[pam[pam_loc]] = le_now
            except Exception as exc:
                print(f"Problem here {exc}")
                continue
        return d

    def is_divergent(self, point, pams, num_points=2000):
        try:
            points = self.sequence_gen_finite(point, pams, num_point=num_points)
            list(points)
            return False
        except:
            return True
