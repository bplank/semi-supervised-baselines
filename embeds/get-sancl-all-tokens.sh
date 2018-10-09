cat ../data/gweb_sancl/pos_fine/*/*  | cut -f1 | sed '/^$/d' > tokens.txt

less ../data/gweb_sancl/unlabeled/gweb-* | sed 's/ /\n/g' | sed '/^$/g' >> tokens.txt
