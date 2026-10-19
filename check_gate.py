import sys; from foodai.foodgate import FoodGate; from PIL import Image; print("food score:", FoodGate().food_score(Image.open(sys.argv[1])))
