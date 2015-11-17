import numpy as np


def similarInputGrids(grid1, grid2):
    # measure how similar the two 2D array inputs are and output
    # this as a percent.
    if (grid1 is not None) and (grid2 is not None):
            totalPrevActiveIns = np.sum(grid1 != 0)
            totalAndGrids = np.sum(np.logical_and(grid2 != 0, grid1 != 0))
            if totalPrevActiveIns > 0:
                percentTemp = float(totalAndGrids) / float(totalPrevActiveIns)
            else:
                # In this case there is no active columns. This means the spatial
                # pooler is not working or the inputs are empty!
                percentTemp = 0
    return percentTemp


def joinInputArrays(input1, input2):
    # Join two input 2D arrays together vstack them.
    # This means the widths of the arrays input1[0] = input2[0]
    # must be equal so one input may need to be padded.
    output = np.array([])
    # check that the inputs are arrays
    assert (type(input1).__name__ == 'ndarray' or
            type(input1).__name__ == 'list')
    assert (type(input2).__name__ == 'ndarray' or
            type(input2).__name__ == 'list')

    if len(input1) > 0 and len(input2) > 0:
        if len(input1[0]) > len(input2[0]):
            # Since input2 is smaller we will pad the array with zeros.
            pad = np.array([0])
            for x in range(len(input1[0]) - len(input2[0]) - 1):
                pad = np.append(pad, [0])
            pad1 = pad
            for y in range(len(input2)-1):
                pad = np.vstack([pad, pad1])
            # Now add the padding to input2
            input2 = np.hstack([input2, pad])
        elif len(input2[0]) > len(input1[0]):
            # Since input1 is smaller we will pad the array with zeros.
            pad = np.array([0])
            for x in range(len(input2[0]) - len(input1[0]) - 1):
                pad = np.append(pad, [0])
            pad1 = pad
            for y in range(len(input2)-1):
                pad = np.vstack([pad, pad1])
            # Now add the padding to input1
            input1 = np.hstack([input1, pad])
        # The arrays should be the same size so now we can vstack them.
        if len(input1[0]) == len(input2[0]):
            # The input arrays have the same width so
            # they can directly be vstacked.
            output = np.vstack([input1, input2])
    elif len(input1) == 0 and len(input2) > 0:
        output = input2
    elif len(input2) == 0 and len(input1) > 0:
        output = input1
    else:
        output = np.array([])
    return output


def checkArraySizesMatch(array1, array2):
        # Check if arrays up to dimension N are of equal size
        subArray1 = array1
        subArray2 = array2
        while (type(subArray1).__name__ == 'ndarray' and
               type(subArray2).__name__ == 'ndarray'):
            if (len(subArray1) != len(subArray2)):
                # arrays are not equal size
                return False
            if ((len(subArray1) == 0 and len(subArray2) != 0) or (
               len(subArray2) == 0 and len(subArray1) != 0)):
                # arrays are not equal size
                return False
            if (len(subArray1) == 0 and len(subArray2) == 0):
                # arrays are equal break to return true
                break
            # Check that the next dimension is still an array
            if (type(subArray1[0]).__name__ == 'ndarray' and
               type(subArray2[0]).__name__ == 'ndarray'):
                subArray1 = subArray1[0]
                subArray2 = subArray2[0]
            else:
                break
        return True


def returnBlankSDRGrid(width, height):
        blankSDR = np.array([[0 for i in range(width)] for j in range(height)])
        return blankSDR

