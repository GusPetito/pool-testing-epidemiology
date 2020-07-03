import matplotlib.pyplot as plt
import numpy as np

virus_rate = .09
population = 22000
iterations = 3
group_sizes = np.arange(2, 61)


def generate_students():
    return np.random.choice([True, False], size=population, p=[virus_rate, 1-virus_rate])


# Each row is a randomly generated list of students
students_expanded = np.array([generate_students() for _ in range(iterations)])


def pool_testing(student_row):
    total_sums = []
    for group_size in group_sizes:
        group_tests = np.add.reduceat(student_row, np.arange(0, population, group_size)).astype(bool)
        total = (group_tests.size) + (group_tests[group_tests].size*group_size)
        total_sums.append(total)
    return total_sums


def grid_testing(student_row):
    total_sum = []
    for group_size in group_sizes:
        total_group_size = group_size**2
        # Adding the people who don't fit into groups
        total = population % total_group_size
        remaining_students = student_row[total:]
        # split_groups.shape => (total group size, number of groups)
        split_groups = np.array(np.split(remaining_students, total_group_size))
        # split_groups.shape => (group length, group length, number of groups)
        split_groups = split_groups.reshape((group_size, group_size, int(remaining_students.size/total_group_size)))

        # +---+---+---+          +---+---+---+
        # | T | F | F |          | T | T | F |
        # +---+---+---+          +---+---+---+
        # | T | F | F |     =>  ...
        # +---+---+---+
        # | F | T | T |
        # +---+---+---+ ...
        # For every group, this reduces each column to a boolean indicating whether or not it's an infected column
        split_groups_columns = split_groups.sum(axis=1).astype(bool)
        # +---+---+---+         +---+
        # | T | T | F |     =>  | 2 |...
        # +---+---+---+         +---+
        # ...
        #For every group, this reduces it down into the number of infected columns
        split_groups_columns = split_groups_columns.sum(axis=0)

        #Similar to above, but for rows
        split_groups_rows = split_groups.sum(axis=1).astype(bool)
        split_groups_rows = split_groups_rows.sum(axis=0)

        # Adding the individual tests needed to be done, after looking at row/column intersections
        total += (split_groups_columns * split_groups_rows).sum()

        # Adding the tests needed to do on each group (each row and column)
        total += int(population / group_size) * 2

        total_sum.append(total)
    return total_sum


# Each row will be a list of y-points (tests done for a specific group size). We then take the average of the columns
# to create one graph
results_grid = np.apply_along_axis(grid_testing, axis=1, arr=students_expanded).mean(axis=0)
results = np.apply_along_axis(pool_testing, axis=1, arr=students_expanded).mean(axis=0)

print(f'''Virus rate: {virus_rate * 100}%
Population: {population}
Minimum tests for pool testing: {int(results[results.argmin()])} tests at a group size of {
int(group_sizes[results.argmin()])}
Minimum tests for grid testing: {int(results_grid[results_grid.argmin()])} tests at a group size of {
int(group_sizes[results_grid.argmin()])}''')

plt.plot(group_sizes, results, label="Pool testing")
plt.plot(group_sizes, results_grid, label="Grid testing")
plt.plot(group_sizes, np.ones(group_sizes.size)*population, label="Individual testing")
plt.ylim(bottom=0)
plt.xlabel("Group size")
plt.ylabel("Tests done")
plt.title("Pool Testing")
plt.legend()
plt.show()