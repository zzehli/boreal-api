from enum import Enum

AERO = ["aero"]
NESCAFE = ["nescafe"]
NATURES_BOUNTY = ["natures-bounty"]
KITKAT = ["kitkat", "kit-kat"]
HAAGEN_DAZS = ["haagen-dazs", "hd"]
STARBUCKS = ["starbucks", "sb"]

class Categories(str, Enum):
    """Document categories."""
    OTHERS = "others"
    PRODUCT = "product"
    RECIPE = "recipe"
    NEWS = "news"
    BLOG = "blog"

if __name__ == "__main__":
    print(Categories.BRAND == "brand")