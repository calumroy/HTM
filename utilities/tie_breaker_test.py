import numpy as np

def make_tie(width, height):
    numColumns = width * height
    normValue = 1.0/float(2*numColumns+2)
    tieBreaker = np.array([[(1+i+j*width)*normValue for i in range(width)] for j in range(height)])
    # Create a tiebreaker that is not biased to either side of the columns grid.
    for j in range(len(tieBreaker)):
        for i in range(len(tieBreaker[0])):
            if (j % 2) == 1:
                # if (i+j*width) % 2 == 1:
                # For odd positions bias to the bottom left
                tieBreaker[j][i] = ((j+1)*width+(width-i-1))*normValue
                # else:
                # For even positions bias to the top left
                #tieBreaker[j][i] = ((width-i-1)+(height-j)*width)*normValue
            else:
                # if (i+j*width) % 2 == 1:
                #     # For odd positions bias to the top right
                #     tieBreaker[j][i] = ((height-j)*width+i)*normValue
                # else:
                # For even positions bias to the bottom right
                tieBreaker[j][i] = (1+i+j*width)*normValue
    return tieBreaker

if __name__ == '__main__':
    width = 5
    height = 10
    tieBreaker = make_tie(width, height)
    print tieBreaker
