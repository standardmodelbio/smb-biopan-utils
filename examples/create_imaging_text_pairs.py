import pandas as pd

from smb_biopan_utils import create_imaging_text_pairs


if __name__ == "__main__":
    file_path = "/path/to/text"
    image_url = "/path/to/images"
    df = pd.read_csv(file_path)
    df["image_url"] = df["impression_id"].apply(lambda x: f"{image_url}/{x}.nii.gz")

    result = create_imaging_text_pairs(
        df,
        image_url_col="image_url",
        output_path="training_data.jsonl",
    )
    print(result)
