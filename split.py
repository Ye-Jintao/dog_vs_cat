import os
import shutil
import argparse
from sklearn.model_selection import train_test_split


def split_dataset(dataset_dir, output_dir, test_size=0.2, val_size=0.1, seed=42):
    """
    将数据集划分为训练集、验证集和测试集
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # 获取所有图像文件
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    cat_files = [f for f in all_files if 'cat' in f.lower()]
    dog_files = [f for f in all_files if 'dog' in f.lower()]

    print(f"Found {len(cat_files)} cat images and {len(dog_files)} dog images")

    # 划分猫图像
    cat_train, cat_test = train_test_split(cat_files, test_size=test_size, random_state=seed)
    cat_train, cat_val = train_test_split(cat_train, test_size=val_size / (1 - test_size), random_state=seed)

    # 划分狗图像
    dog_train, dog_test = train_test_split(dog_files, test_size=test_size, random_state=seed)
    dog_train, dog_val = train_test_split(dog_train, test_size=val_size / (1 - test_size), random_state=seed)

    # 复制文件到相应目录
    def copy_files(files, category, split):
        for f in files:
            src = os.path.join(dataset_dir, f)
            if split == 'test':
                dst = os.path.join(output_dir, split, f)
            else:
                dst = os.path.join(output_dir, split, category, f)
            shutil.copy2(src, dst)

    # 复制猫图像
    copy_files(cat_train, 'cat', 'train')
    copy_files(cat_val, 'cat', 'val')
    copy_files(cat_test, 'cat', 'test')

    # 复制狗图像
    copy_files(dog_train, 'dog', 'train')
    copy_files(dog_val, 'dog', 'val')
    copy_files(dog_test, 'dog', 'test')

    print(f"Dataset split complete:")
    print(f"  Training set: {len(cat_train) + len(dog_train)} images")
    print(f"  Validation set: {len(cat_val) + len(dog_val)} images")
    print(f"  Test set: {len(cat_test) + len(dog_test)} images")

    # 创建标签文件
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Training set: {len(cat_train) + len(dog_train)} images\n")
        f.write(f"Validation set: {len(cat_val) + len(dog_val)} images\n")
        f.write(f"Test set: {len(cat_test) + len(dog_test)} images\n")
        f.write(f"Cat images: {len(cat_files)}\n")
        f.write(f"Dog images: {len(dog_files)}\n")
        f.write(f"Split seed: {seed}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Dogs vs Cats dataset into train, val and test sets')
    parser.add_argument('--dataset_dir', type=str, default='./train',
                        help='Directory containing all images (default: ./train)')
    parser.add_argument('--output_dir', type=str, default='./split_data',
                        help='Output directory for split datasets (default: ./split_data)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion for test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion for validation set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print("Starting dataset split...")
    split_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )
    print("Dataset split completed successfully!")