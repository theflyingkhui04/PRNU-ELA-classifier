from PIL import Image

def enhance_brightness(img: Image.Image, scale: float) -> Image.Image:
    
    width, height = img.size
    pixels = img.load()
    
    bright_img = Image.new("RGB", (width, height))
    bright_pixels = bright_img.load()
    
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            r = int(min(255, r * scale))
            g = int(min(255, g * scale))
            b = int(min(255, b * scale))
            bright_pixels[x, y] = (r, g, b)

    return bright_img


def getextrema(img: Image.Image):
    
    img = img.convert("RGB")
    width, height = img.size
    pixels = img.load()

    rmin, rmax = 255, 0
    gmin, gmax = 255, 0
    bmin, bmax = 255, 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            rmin = min(rmin, r)
            rmax = max(rmax, r)
            gmin = min(gmin, g)
            gmax = max(gmax, g)
            bmin = min(bmin, b)
            bmax = max(bmax, b)

    return (rmin, rmax), (gmin, gmax), (bmin, bmax)



def convert_to_ela(image_path, quality=90):
    original = image_path.convert("RGB")
    temp_file = 'temp_ela.jpg'
    original.save(temp_file, "JPEG", quality=quality)
    compressed = Image.open(temp_file).convert("RGB")
    diff = Image.new("RGB", original.size)

   
    width, height = original.size
    for x in range(width):
        for y in range(height):
            r1, g1, b1 = original.getpixel((x, y))
            r2, g2, b2 = compressed.getpixel((x, y))
            diff_pixel = (
                abs(r1 - r2),
                abs(g1 - g2),
                abs(b1 - b2)
            )
            diff.putpixel((x, y), diff_pixel)
    extrema = getextrema(diff)
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    diff = enhance_brightness(diff, scale)

    return diff
    
