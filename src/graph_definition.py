from PIL import Image

img = Image.open("graph.png").convert("RGBA")

data = img.getdata()

new_data = []

for item in data:
    if item[0] > 245 and item[1] > 245 and item[2] > 245:
        new_data.append((255, 255, 255, 0))
    else:
        new_data.append(item)

img.putdata(new_data)

img.save("graph_transparent.png")