python main.py --input_dir=../../data/vqa/inputs224 --skip_stage2 --no_pretrain_enc --qst_only --num_epochs=20 --arch_type=fixed-darts --batch_size=64  --exp=qst_only_darts
python main.py --input_dir=../../data/vqa/inputs224 --skip_stage2 --qst_only --num_epochs=20 --arch_type=fixed-darts --batch_size=64  --exp=qst_only_darts_pretrain
