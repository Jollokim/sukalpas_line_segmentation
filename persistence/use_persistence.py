import numpy as np
from persistence.persistence1d import RunPersistence

def main():
    #~ Set the data directly in code
    InputData = np.array([2.0, 5.0, 7.0, -12.0, -13.0, -7.0, 10.0, 18.0, 6.0, 8.0, 7.0, 4.0])

    # #~ This simple call is all you need to compute the extrema of the given data and their persistence.
    ExtremaAndPersistence = RunPersistence(InputData)

    # #~ Keep only those extrema with a persistence larger than 10.
    Filtered = [t for t in ExtremaAndPersistence if t[1] > 10]

    # print(ExtremaAndPersistence)

    # #~ Sort the list of extrema by index.
    Sorted = sorted(Filtered, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[1])

    #~ Print to console
    for i, E in enumerate(Sorted):
        strType = "Minimum" if (i % 2 == 0) else "Maximum"
        print("%s at index %d with persistence %g and data value %g" % (strType, E[0], E[1], InputData[E[0]]))

def get_persistence(input, persistence):
    #~ This simple call is all you need to compute the extrema of the given data and their persistence.
    ExtremaAndPersistence = RunPersistence(input)

    # print(ExtremaAndPersistence)
    # print()

    #~ Keep only those extrema with a persistence larger than 10.
    Filtered = [t for t in ExtremaAndPersistence if t[1] > persistence]

    # print(Filtered)
    # print()

    #~ Sort the list of extrema by persistence.
    Sorted = sorted(Filtered, key=lambda ExtremumAndPersistence: ExtremumAndPersistence[0])

    # print(Sorted)

    return Sorted

if __name__ == '__main__':
    main()