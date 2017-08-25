""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    x_min = min(arr)
    x_max = max(arr)
    print(x_min)
    print(x_max)
    normalized_arr = []
    for x in arr:
        x_norm = (x-x_min)/float((x_max-x_min))
        normalized_arr.append(x_norm)
    return normalized_arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print(featureScaling(data))
