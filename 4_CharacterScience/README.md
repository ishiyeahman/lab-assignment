# フォントの印象分類
特定の印象語に対し，フォント画像のOne-vs-Restの2クラス分類を行う．

## 実行方法
1. 以下のリンクから本ディレクトリにデータセットをダウンロード<br>https://www.cs.rochester.edu/u/tchen45/font/font.html
2. Dockerfileに従って環境構築
3. preprocess_data.pyを実行
4. main.pyを実行

## データセットに関する注意事項
- 上記のリンクからダウンロードしたデータセットでは，taglabelフォルダにblank fileがあるため，taglabel.zipを解凍したものと置き換える
- data/1vsO/..にいくつかの印象語について分類を行った学習過程・結果のデータあり
