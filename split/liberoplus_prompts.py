"""Instructions for LIBERO-Plus tasks, as the LIBERO-trained models expect them.

LIBERO-Plus builds task.language from each variant's file name, so most variants get a tag appended ("... on the plate
view 0 0 100 0 0 initstate 196"; LIBERO-plus issues #64, #65). The BDDL (:language ...) text differs from the training
instruction too. So we strip the tag and rebuild the instruction as LIBERO does (grab_language_from_filename); language
variants keep LIBERO-Plus's rewritten instruction, which is the perturbation.
"""

import re

# Anchored at the end: base names contain words like "table" ("from_table_center")
TAGS = [
    r"_(table|tb)_\d+$",                                          # background textures
    r"_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+(_noise_\d+)?$",     # camera, robot initial state, sensor noise
    r"_add_\d+$", r"_(moved_)?level\d+_sample\d+$",               # objects layout
    r"_light_\d+$",                                               # light
]
LANGUAGE_TAG = r"_language_\d+(_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+)?$"


def base_task_name(name):
    for tag in [LANGUAGE_TAG] + TAGS:  # language first: its variants also end in a view tag
        stripped = re.sub(tag, "", name)
        if stripped != name:
            return stripped
    return name


def libero_language(base_name):
    """LIBERO's grab_language_from_filename."""
    if base_name[0].isupper():  # LIBERO-10 / LIBERO-90: scene prefix
        return " ".join(base_name[base_name.find("SCENE") + (8 if "SCENE10" in base_name else 7):].split("_"))
    return " ".join(base_name.split("_"))


def instruction(task):
    if re.search(LANGUAGE_TAG, task.name):
        return task.language  # read from the variant's BDDL: the rewritten instruction
    return libero_language(base_task_name(task.name))
