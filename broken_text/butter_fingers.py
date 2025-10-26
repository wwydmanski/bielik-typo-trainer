import itertools
import random

"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


def _butter_finger(text, prob=0.1, keyboard="qwerty", seed=0, max_outputs=1):
    random.seed(seed)
    key_approx = {}

    if keyboard == "qwerty":
        key_approx["q"] = "qwasedzx"
        key_approx["w"] = "wqesadrfcx"
        key_approx["e"] = "ewrsfdqazxcvgt"
        key_approx["r"] = "retdgfwsxcvgt"
        key_approx["t"] = "tryfhgedcvbnju"
        key_approx["y"] = "ytugjhrfvbnji"
        key_approx["u"] = "uyihkjtgbnmlo"
        key_approx["i"] = "iuojlkyhnmlp"
        key_approx["o"] = "oipklujm"
        key_approx["p"] = "plo['ik"

        key_approx["a"] = "aqszwxwdce"
        key_approx["s"] = "swxadrfv"
        key_approx["d"] = "decsfaqgbv"
        key_approx["f"] = "fdgrvwsxyhn"
        key_approx["g"] = "gtbfhedcyjn"
        key_approx["h"] = "hyngjfrvkim"
        key_approx["j"] = "jhknugtblom"
        key_approx["k"] = "kjlinyhn"
        key_approx["l"] = "lokmpujn"

        key_approx["z"] = "zaxsvde"
        key_approx["x"] = "xzcsdbvfrewq"
        key_approx["c"] = "cxvdfzswergb"
        key_approx["v"] = "vcfbgxdertyn"
        key_approx["b"] = "bvnghcftyun"
        key_approx["n"] = "nbmhjvgtuik"
        key_approx["m"] = "mnkjloik"
        key_approx[" "] = " "
    else:
        print("Keyboard not supported.")

    prob_of_typo = int(prob * 100)
    perturbed_texts = []
    for _ in itertools.repeat(None, max_outputs):
        butter_text = ""
        for letter in text:
            lcletter = letter.lower()
            if lcletter not in key_approx.keys():
                new_letter = lcletter
            else:
                if random.choice(range(0, 100)) <= prob_of_typo:
                    new_letter = random.choice(key_approx[lcletter])
                else:
                    new_letter = lcletter
            # go back to original case
            if not lcletter == letter:
                new_letter = new_letter.upper()
            butter_text += new_letter
        perturbed_texts.append(butter_text)
    return perturbed_texts