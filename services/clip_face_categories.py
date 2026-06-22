# services/clip_face_categories.py

HUMAN_IDENTIFIABLE_PROMPTS = [
    "a clear real human face",
    "a real identifiable human face",
    "a natural human face in video footage",
    "a real person looking at the camera",
    "a real person in a news video",
    "a real human portrait",
    "a clear face of a real living person",
    "a real adult human face",
    "a real elderly human face",
    "a real human face with glasses",
    "a real human face with beard",
    "a real human face with mustache",
    "a real human face in archival video",
    "a real human face in television footage",
    "a real politician face on television",
    "a real presenter face in a broadcast",
]


HUMAN_UNIDENTIFIABLE_PROMPTS = [
    "a real person with a covered face",
    "a human face covered by a mask",
    "a person wearing a medical mask",
    "a person wearing a balaclava",
    "a person wearing a gas mask",
    "a person wearing a helmet visor",
    "a person with face hidden by scarf",
    "a person with face partially covered",
    "a blurry human face",
    "a very small human face",
    "a tiny face in the background",
    "a distant face too small to identify",
    "a very small background face",
    "a face far away in a crowd",
    "a partially visible human face",
    "a cropped human face",
    "a side profile human face difficult to identify",
    "a human face hidden in shadow",
    "a person with sunglasses and mask",
    "a face blocked by hand",
    "a person with face obscured",
]


NON_HUMAN_PROMPTS = [
    "an animal face",
    "a dog face",
    "a cat face",
    "a horse face",
    "a bear face",
    "a monkey face",
    "a toy animal face",
    "a face-like pattern on an object",
    "an object that looks like a face",
    "a face-like shape on food",
    "a face-like pattern on fabric",
    "a face-like image on a package",
    "a face-like pattern on a wall",
    "a face-like pattern in artwork",
    "a face-like image on a wall decoration",
    "a face-like decorative object",
    "a face-like object",
    "not a human face",
    "a false face detection",
]


ARTIFICIAL_HUMAN_PROMPTS = [
    # mannequin / doll / puppet
    "a doll face",
    "a puppet face",
    "a mannequin face",
    "a shop mannequin face",
    "a museum mannequin face",
    "a historical mannequin face",
    "a life size mannequin",
    "a plastic human face",
    "a toy human face",

    # wax figure / statue / sculpture
    "a wax figure face",
    "a museum wax figure",
    "a statue face",
    "a statue human face",
    "a face of a statue",
    "a face of a statue in the background",
    "a small statue human face",
    "a decorative statue face",
    "a decorative statue face in a room",
    "a human statue face",
    "a sculpture face",
    "a sculpture of a human face",
    "a human sculpture",
    "a bust sculpture face",
    "a human bust statue",
    "a stone statue face",
    "a marble statue face",
    "a bronze statue face",
    "a carved human face",
    "a religious statue face",

    # museum / historical exhibits
    "a museum exhibit with a human face",
    "a historical exhibit with a human face",
    "a reconstructed historical person",
    "a museum display figure",
    "a human figure in a museum display",
    "a historical costume mannequin",
    "a museum doll",

    # paintings / drawings / portraits
    "a painted human face",
    "a painted face",
    "a face in a painting",
    "a portrait painting",
    "a historical portrait painting",
    "an oil painting portrait",
    "a painted portrait on canvas",
    "a painted face on canvas",
    "a drawn human face",
    "an illustrated human face",
    "a sketch of a human face",
    "an artwork depicting a person",
    "a face in artwork",
    "a decorative human face",
    "a decorative portrait",

    # framed portraits / framed photos / wall art
    "a face in a framed portrait",
    "a human face in a framed picture on a wall",
    "a portrait hanging on a wall",
    "a face in a framed photograph",
    "a framed portrait in the background",
    "a face in wall art",
    "a mural face",
    "a decorative picture with a human face",

    # religious icons / religious paintings
    "a face in a religious icon",
    "a painted religious figure",
    "a saint portrait in an icon",
    "a religious painting with a human face",
    "a painted icon of a saint",
    "a human face in an icon",

    # tapestry / textile / embroidery
    "a face in a tapestry",
    "a tapestry face",
    "a face on a tapestry",
    "a woven human face",
    "a woven face on fabric",
    "an embroidered face",
    "an embroidered human face",
    "a decorative textile face",

    # cartoon / animation / rendered
    "a cartoon human face",
    "an animated character face",
    "a 3d rendered human-like face",
    "a computer generated human-like face",

    # masks
    "a mask shaped like a human face",
    "a decorative mask",
    "a human face mask",
]


AI_GENERATED_PROMPTS = [
    "an ai generated human face",
    "a synthetic human face",
    "a 3d rendered human face",
    "a computer generated human face",
    "a digital avatar face",
    "a realistic artificial human face",
    "a deepfake human face",
    "a virtual human face",
    "a CGI human face",
]


UNCERTAIN_PROMPTS = [
    "an unclear face",
    "a low quality face image",
    "a confusing face-like image",
    "an ambiguous face detection",
    "a distorted face",
    "a noisy video frame with a face-like region",
    "a heavily compressed video face",
    "an uncertain face crop",
]


CLIP_FACE_CATEGORIES = {
    "real_identifiable": HUMAN_IDENTIFIABLE_PROMPTS,
    "real_unidentifiable": HUMAN_UNIDENTIFIABLE_PROMPTS,
    "non_human": NON_HUMAN_PROMPTS,
    "artificial_human": ARTIFICIAL_HUMAN_PROMPTS,
    "ai_generated": AI_GENERATED_PROMPTS,
    "uncertain": UNCERTAIN_PROMPTS,
}


HUMAN_CATEGORIES = {
    "real_identifiable",
    "real_unidentifiable",
    "artificial_human",
    "ai_generated",
}


IDENTIFIABLE_CATEGORIES = {
    "real_identifiable",
}


NON_IDENTIFIABLE_CATEGORIES = {
    "real_unidentifiable",
    "non_human",
    "artificial_human",
    "ai_generated",
    "uncertain",
}
