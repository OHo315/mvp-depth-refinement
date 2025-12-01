#DIODE_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/diode_depth"
#DIODE_TXT_FILEPATH="$DIODE_TXT_DIRPATH/diode_train_indoor_filename_list.txt"
#mkdir -p $DIODE_TXT_DIRPATH
#find $BASE_DATA_DIR/diode -type f -name "*png" > $DIODE_TXT_FILEPATH
#sed -i "s|$BASE_DATA_DIR/||g" $DIODE_TXT_FILEPATH

#ARKIT_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/arkit"
#ARKIT_TXT_FILEPATH="$ARKIT_TXT_DIRPATH/arkit_train_filename_list.txt"
#mkdir -p $ARKIT_TXT_DIRPATH
#find $BASE_DATA_DIR/arkitscenes_processed -type f -name "*png" > $ARKIT_TXT_FILEPATH
#sed -i "s|$BASE_DATA_DIR/||g" $ARKIT_TXT_FILEPATH

WAYMO_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/waymo"
WAYMO_TXT_FILEPATH="$WAYMO_TXT_DIRPATH/waymo_train_filename_list.txt"
mkdir -p $WAYMO_TXT_DIRPATH
find $BASE_DATA_DIR/waymo_preprocess -type f -name "*png" > $WAYMO_TXT_FILEPATH
sed -i "s|$BASE_DATA_DIR/||g" $WAYMO_TXT_FILEPATH
