from smb_biopan_utils import process_mm_info


if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data/test.nii.gz"
                }
            ]
        }
    ]
    print(process_mm_info(messages))