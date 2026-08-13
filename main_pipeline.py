"""
main_pipeline.py
Integra os três módulos (equivalente ao "main vision module" do artigo e roda um teste com uma imagem real.
Uso:
    python main_pipeline.py

Por padrão usa DEFAULT_COLOR_SPACES (valores de exemplo em color_common.py).
Para usar valores reais calibrados na câmera, troque a linha marcada abaixo
para chamar load_color_spaces_from_csv("meus_valores_hsv.csv").
"""

import cv2

from color_common import DEFAULT_COLOR_SPACES, build_color_detector, load_color_spaces_from_csv
from field_boundary_detector import detect_field_boundary
from obstacle_detector import detect_obstacles
from line_detector import detect_line_points

IMAGE_PATH = "images/bitbots_reality_spl_only_131-16_02_2018__11_18_08_0117_upper_png_jpg_b0.8_s1.0_k0.jpg"

# --------------------------------------------------------------------------
# Troque para valores reais aqui quando tiver o CSV calibrado:
#
#   color_spaces = load_color_spaces_from_csv("meus_valores_hsv.csv")
#
# (a leitura do pd.read_csv acontece dentro de load_color_spaces_from_csv,
#  em color_common.py)
# --------------------------------------------------------------------------
color_spaces = DEFAULT_COLOR_SPACES
color_detector = build_color_detector(color_spaces)


def run(image_bgr, head_tilted_up: bool = False):
    boundary_points, hull_mask = detect_field_boundary(
        image_bgr, color_detector, search_from_top=not head_tilted_up
    )
    obstacles = detect_obstacles(image_bgr, color_detector, boundary_points, hull_mask)
    line_points = detect_line_points(image_bgr, color_detector, boundary_points)

    return {
        "field_boundary": boundary_points,
        "hull_mask": hull_mask,
        "obstacles": obstacles,
        "line_points": line_points,
    }


def draw_debug_image(image_bgr, result):
    """Gera uma imagem de debug similar à Fig. 1 do artigo."""
    debug = image_bgr.copy()

    for x, y in result["field_boundary"]:
        cv2.circle(debug, (x, y), 2, (0, 0, 255), -1)  # linha vermelha (pontos)

    color_map = {"red": (0, 0, 255), "blue": (255, 0, 0), "white": (255, 255, 255), "unknown": (0, 255, 255)}
    for obs in result["obstacles"]:
        color = color_map.get(obs.color_class, (0, 255, 255))
        cv2.rectangle(debug, (obs.x, obs.y), (obs.x + obs.width, obs.y + obs.height), color, 2)

    for x, y in result["line_points"]:
        cv2.circle(debug, (x, y), 1, (0, 0, 255), -1)

    return debug


if __name__ == "__main__":
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Não consegui abrir a imagem: {IMAGE_PATH}")

    result = run(image)

    print(f"Pontos de fronteira do campo: {len(result['field_boundary'])}")
    print(f"Obstáculos detectados: {len(result['obstacles'])}")
    for obs in result["obstacles"]:
        print(f"  - {obs.color_class} em {obs.bbox}")
    print(f"Pontos de linha detectados: {len(result['line_points'])}")

    debug_img = draw_debug_image(image, result)
    cv2.imwrite("debug_output.jpg", debug_img)
    print("Imagem de debug salva em debug_output.jpg")
