DIODE_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/diode_depth"
DIODE_TXT_FILEPATH="$DIODE_TXT_DIRPATH/diode_train_indoor_filename_list.txt"
DIODE_DATADIR=$BASE_DATA_DIR/diode
mkdir -p $DIODE_TXT_DIRPATH
find $DIODE_DATADIR -type f -name "*png" > $DIODE_TXT_FILEPATH
sed -i "s|$DIODE_DATADIR/||g" $DIODE_TXT_FILEPATH

ARKIT_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/arkit_depth"
ARKIT_TXT_FILEPATH="$ARKIT_TXT_DIRPATH/arkit_train_filename_list.txt"
ARKIT_DATADIR=$BASE_DATA_DIR/arkitscenes_processed
mkdir -p $ARKIT_TXT_DIRPATH
find $ARKIT_DATADIR -type f -name "*png" > $ARKIT_TXT_FILEPATH
sed -i "s|$ARKIT_DATADIR/||g" $ARKIT_TXT_FILEPATH

WAYMO_TXT_DIRPATH="$BASE_DATA_DIR/../data_split/waymo_depth"
WAYMO_TXT_FILEPATH="$WAYMO_TXT_DIRPATH/waymo_train_filename_list.txt"
WAYMO_DATADIR=$BASE_DATA_DIR/waymo_preprocess
mkdir -p $WAYMO_TXT_DIRPATH
find $WAYMO_DATADIR -type f -name "*png" > $WAYMO_TXT_FILEPATH
sed -i "s|$WAYMO_DATADIR/||g" $WAYMO_TXT_FILEPATH
