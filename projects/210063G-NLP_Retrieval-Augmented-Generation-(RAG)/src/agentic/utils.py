def dict_to_str(data: dict) -> str:
    def section_to_str(section_name, items):
        # If empty, show section title with no items
        if not items:
            return f"--{section_name}--\n(No {section_name})\n\n"
        
        section_str = f"--{section_name}--\n"
        for i, item in enumerate(items, 1):
            section_str += f"[{i}] " + "\n".join(
                f"{key}: {value}" for key, value in item.items()
            ) + "\n\n"
        return section_str

    # Build all sections
    parts = []
    for key in ["entities", "relationships", "chunks", "references"]:
        items = data.get(key, [])
        parts.append(section_to_str(key, items))
    
    return "".join(parts)
