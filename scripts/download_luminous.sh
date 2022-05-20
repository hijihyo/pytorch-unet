# 프로젝트 폴더 최상단에서 실행할 것

# Download Luminous database (which is for ultrasound image segmentation)
wget --no-check-certificate https://www.dropbox.com/s/r77rik8jg9qsp4m/LUMINOUS_Database.zip --directory-prefix .data
unzip .data/LUMINOUS_Database.zip -d .data
mv .data/LUMINOUS_Database .data/Luminous
rm .data/LUMINOUS_Database.zip