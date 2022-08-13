import os

xs = (x * 0.1 for x in range(0, 40))
for x in xs:
    start_position = 1.0;
    this_position = start_position + x;
    command = "cargo run --release -- " + str(this_position) + " > output/gif2/frame_" + str(x) + ".ppm"
    os.system(command)