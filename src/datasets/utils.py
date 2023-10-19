from __future__ import annotations


def unlabel_annotation(ann_path: str, default: int, target: int = 0) -> str:
    undef_objs = []
    # Read the file with original annotation
    with open(ann_path, "r", encoding="utf-8") as file:
        objs = file.readlines()
    # Replace the class value with target
    for obj in objs:
        if obj.split()[:1][0] == str(default):
            undef_objs.append(f"{target} {' '.join(obj.split()[1:])}")
    # Rewrite the annotation file
    with open(ann_path, "w", encoding="utf-8") as file:
        file.write("\n".join(undef_objs))

    return ann_pat