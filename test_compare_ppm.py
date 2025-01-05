import pytest
import numpy as np

def read_ppm(file_path: str) -> np.ndarray:
    with open(file_path, "r") as f:
        # Read header
        header = f.readline().strip()
        assert header == "P3", f"Unsupported PPM format: {header}"

        # Read dimensions
        line = f.readline().strip()
        dimensions = line.split()
        width, height = int(dimensions[0]), int(dimensions[1])
        
        # Read max color value
        max_color = int(f.readline().strip())
        assert max_color == 255, f"Unsupported max color value: {max_color}"
        
        # Read pixel data
        pixel_data = f.readlines()
        assert len(pixel_data) == width * height, \
            f"Missmatch in number of pixels={len(pixel_data)}, expected: width * height={width*height}"
        
        data = np.zeros((height, width, 3), dtype=np.uint8)
        for index, pixel in enumerate(pixel_data):
            row = index // width
            col = index % width
            r, g, b = pixel.split()
            data[row, col] = int(r), int(g), int(b)

        return data


def compare_ppm(file1: str, file2: str) -> bool:
    data1 = read_ppm(file1)
    data2 = read_ppm(file2)
    
    np.testing.assert_array_equal(data1, data2)
    return True


def test_compare_ppm():
    reference_file = "fast-math-pic.ppm"
    output_file = "out.ppm"

    assert compare_ppm(reference_file, output_file), "The files do not match."


if __name__ == "__main__":
    pytest.main([__file__])
