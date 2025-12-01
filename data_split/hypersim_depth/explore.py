from tqdm import tqdm

if __name__ == "__main__":
    ans = 0
    with open("filename_list_train_filtered.txt", "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for line in tqdm(lines):
            if int(line.split("_")[1]) < 30:
                ans += 1
    print(ans)

        
