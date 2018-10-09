import numpy

import util
PARETO_OPTIMAL_VALUES = util.load_result_file("question_2_data_preparation_3.npz")["pareto_weights"]
OPTIMAL_VALUE = 20 # SET THIS VALUE

INPUT_WEIGHTS = PARETO_OPTIMAL_VALUES[OPTIMAL_VALUE][0]
FC_BIAS = PARETO_OPTIMAL_VALUES[OPTIMAL_VALUE][1]
FC_WEIGHTS = PARETO_OPTIMAL_VALUES[OPTIMAL_VALUE][2]
OUTPUT_BIAS = PARETO_OPTIMAL_VALUES[OPTIMAL_VALUE][3]

VALUES = ["INPUT_WEIGHTS", "FC_BIAS", "FC_WEIGHTS", "OUTPUT_BIAS"]

def get_value(value):
    return globals()[value]

class NeuralNetwork:
    def floating_point(self, value):
        casted_values = getattr(numpy, self.word_width)(value)
        self._record_range(casted_values)
        return casted_values

    def __init__(self, word_width):
        self.word_width = word_width
        self.upper_bound = 0
        self.lower_bound = 0

    def __reLU(self, biased_input):
        return max(0, biased_input)

    def run(self, picture_8x8):
        input_picture = picture_8x8.flatten()
        data_value = input_picture

        for layer in range(0, len(VALUES), 2):
            data_value = self.floating_point(data_value)
            weights = self.floating_point(get_value(VALUES[layer]))

            # computation
            input_to_activation = numpy.dot(data_value, weights)

            #input to the next layer
            input_to_activation = self.floating_point(input_to_activation)
            bias = self.floating_point(get_value(VALUES[layer+1]))

            # computation
            data_value = self._activate(input_to_activation, bias)

        return data_value

    def _activate(self, input_values, bias):
        biased_inputs = input_values + bias

        outputs = []
        for biased_input in numpy.nditer(biased_inputs):
            outputs.append(self.__reLU(biased_input))

        return numpy.array(outputs, dtype=self.word_width)

    def _record_range(self, casted_values):
        minimum = numpy.amin(casted_values)
        maximum = numpy.amax(casted_values)

        if minimum < self.lower_bound:
            self.lower_bound = minimum

        if maximum > self.upper_bound:
            self.upper_bound = maximum

class Tester:
    def __init__(self):
        self.verbose = True

    def setup(self, max_set):
        if max_set:
            # test against the entire thing damnit
            self.validation_set_x = util.load_result_file("reduced_x.npy")
            self.validation_set_y = util.load_result_file("reduced_y.npy")
        else:
            self.validation_set_x = util.load_result_file("validation_set_x.npy")
            self.validation_set_y = util.load_result_file("validation_set_y.npy")

    def test(self, word_width):
        n = NeuralNetwork(word_width)
        correct_tests = 0

        for test in range(len(self.validation_set_x)):
            predicted_y = n.run(self.validation_set_x[test])
            test_result = self.validate(predicted_y, self.validation_set_y[test], test+1)

            correct_tests = correct_tests + test_result

        print(n.upper_bound, n.lower_bound)
        correct_percentage = 100 * float(correct_tests) / len(self.validation_set_x)
        # print("Overall correct percentage is %s" % (correct_percentage))
        return correct_percentage

    def validate(self, predicted_value, real_value, iteration_number):
        predicted_value_list = predicted_value.tolist()
        real_value_list = real_value.tolist()

        predicted_value = predicted_value_list.index(max(predicted_value_list))
        real_value = real_value_list.index(max(real_value_list))

        if real_value == predicted_value:
            if self.verbose: print("SUCCESS!")
            return 1
        else:
            if self.verbose: print("FAILED, the correct answer was %s, you guessed %s" %(real_value, predicted_value))
            return 0

    def graph_float(self):
        self.verbose = False

        word_widths = ["float16"]

        with_validation_data = []
        with_full_data = []

        # populate validation_data
        self.setup(False)
        for word_width in word_widths:
            percentage = self.test(word_width)
            print("Testing with " + word_width + " yielded " + str(float(percentage)) + " of success")
            with_validation_data.append(percentage)

        # with full data
        self.setup(True)
        for word_width in word_widths:
            percentage = self.test(word_width)
            print("Testing with " + word_width + " yielded " + str(float(percentage)) + " of success with the full set")
            with_full_data.append(percentage)

        print(with_validation_data, with_full_data)
        return word_widths, with_validation_data, with_full_data

Tester().graph_float()
