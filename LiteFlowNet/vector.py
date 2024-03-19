import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv("../vectors_bounding_box.csv")

def read_nth_line(filename, n):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == n - 1:
                line = line.split()
                x = (int(line[0])+int(line[2]))/2
                y = (int(line[1])+int(line[3]))/2
                return x, y
    return None

prev_frame_num = 0
first = True
print("Total:", len(df))
for i in range(len(df)):
    if i%50==0:
        print(i)
    frame_num = int(df.iloc[i][0])
    if prev_frame_num!=frame_num:
        if not first:
            plt.savefig("../Frames_with_vectors/Frame_{}.png".format(prev_frame_num))
        plt.figure()
        prev_frame_num = frame_num
        first = False
    line_num = df.iloc[i][1]
    u = int(df.iloc[i][2])/100
    v = int(df.iloc[i][3])/100
    x, y = read_nth_line("../coordinates_bounding_box/Frame_{}.txt".format(frame_num), line_num)
    image = mpimg.imread("../Frames/Frame_{}.png".format(frame_num))
    plt.imshow(image)
    plt.quiver(x, y, u, v, scale=12)