cd ..
rm Docs
mkdir Docs
cd Docs
sphinx-apidoc -F -H 'BatchRL' -A 'Chris' -o . '../MasterThesis/BatchRL/'
cp ../MasterThesis/conf.py .
python ../MasterThesis/doc_helper.py
./make html SPHINXBUILD='python $(shell which sphinx-build)'
cd ../MasterThesis