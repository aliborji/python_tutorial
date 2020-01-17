document.write('<link rel="stylesheet" href="https://github.githubassets.com/assets/gist-embed-400c181cb6431951ea5479cf30c36933.css">')
document.write('<div id=\"gist97509669\" class=\"gist\">\n    <div class=\"gist-file\">\n      <div class=\"gist-data\">\n        <div class=\"js-gist-file-update-container js-task-list-container file-box\">\n  <div id=\"file-9t_11-py\" class=\"file\">\n    \n\n  <div itemprop=\"text\" class=\"Box-body p-0 blob-wrapper data type-python \">\n      \n<table class=\"highlight tab-size js-file-line-container\" data-tab-size=\"8\">\n      <tr>\n        <td id=\"file-9t_11-py-L1\" class=\"blob-num js-line-number\" data-line-number=\"1\"><\/td>\n        <td id=\"file-9t_11-py-LC1\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># each model is sooo big we can&#39;t fit both in memory<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L2\" class=\"blob-num js-line-number\" data-line-number=\"2\"><\/td>\n        <td id=\"file-9t_11-py-LC2\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>encoder_rnn<\/span>.<span class=pl-en>cuda<\/span>(<span class=pl-c1>0<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L3\" class=\"blob-num js-line-number\" data-line-number=\"3\"><\/td>\n        <td id=\"file-9t_11-py-LC3\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>decoder_rnn<\/span>.<span class=pl-en>cuda<\/span>(<span class=pl-c1>1<\/span>)<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L4\" class=\"blob-num js-line-number\" data-line-number=\"4\"><\/td>\n        <td id=\"file-9t_11-py-LC4\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L5\" class=\"blob-num js-line-number\" data-line-number=\"5\"><\/td>\n        <td id=\"file-9t_11-py-LC5\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># run input through encoder on GPU 0<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L6\" class=\"blob-num js-line-number\" data-line-number=\"6\"><\/td>\n        <td id=\"file-9t_11-py-LC6\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>encoder_out<\/span> <span class=pl-c1>=<\/span> <span class=pl-en>encoder_rnn<\/span>(<span class=pl-s1>x<\/span>.<span class=pl-en>cuda<\/span>(<span class=pl-c1>0<\/span>))<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L7\" class=\"blob-num js-line-number\" data-line-number=\"7\"><\/td>\n        <td id=\"file-9t_11-py-LC7\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L8\" class=\"blob-num js-line-number\" data-line-number=\"8\"><\/td>\n        <td id=\"file-9t_11-py-LC8\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># run output through decoder on the next GPU<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L9\" class=\"blob-num js-line-number\" data-line-number=\"9\"><\/td>\n        <td id=\"file-9t_11-py-LC9\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>out<\/span> <span class=pl-c1>=<\/span> <span class=pl-en>decoder_rnn<\/span>(<span class=pl-s1>encoder_out<\/span>.<span class=pl-en>cuda<\/span>(<span class=pl-c1>1<\/span>))<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L10\" class=\"blob-num js-line-number\" data-line-number=\"10\"><\/td>\n        <td id=\"file-9t_11-py-LC10\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L11\" class=\"blob-num js-line-number\" data-line-number=\"11\"><\/td>\n        <td id=\"file-9t_11-py-LC11\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># normally we want to bring all outputs back to GPU 0<\/span><\/td>\n      <\/tr>\n      <tr>\n        <td id=\"file-9t_11-py-L12\" class=\"blob-num js-line-number\" data-line-number=\"12\"><\/td>\n        <td id=\"file-9t_11-py-LC12\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>out<\/span> <span class=pl-c1>=<\/span> <span class=pl-s1>out<\/span>.<span class=pl-en>cuda<\/span>(<span class=pl-c1>0<\/span>)<\/td>\n      <\/tr>\n<\/table>\n\n\n  <\/div>\n\n  <\/div>\n<\/div>\n\n      <\/div>\n      <div class=\"gist-meta\">\n        <a href=\"https://gist.github.com/williamFalcon/296fa0b0605c7e6052d74f89a7d39d68/raw/5a6d16675daf97acf65ebffba2224df3aec29b4c/9t_11.py\" style=\"float:right\">view raw<\/a>\n        <a href=\"https://gist.github.com/williamFalcon/296fa0b0605c7e6052d74f89a7d39d68#file-9t_11-py\">9t_11.py<\/a>\n        hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n      <\/div>\n    <\/div>\n<\/div>\n')