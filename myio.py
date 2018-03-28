# -*- coding: utf-8 -*-
#!/usr/bin/env python

import codecs
import os
import pickle

def conv_encoding(data, to_enc="utf-8"):
    """
    stringのエンコーディングを変換する
    @param ``data'' str object.
    @param ``to_enc'' specified convert encoding.
    @return str object.
    """
    lookup = ('utf_8', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213',
            'shift_jis', 'shift_jis_2004','shift_jisx0213',
            'iso2022jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_3',
            'iso2022_jp_ext','latin_1', 'ascii')
    for encoding in lookup:
        try:
            data = data.decode(encoding)
            break
        except:
            pass
    if isinstance(data, unicode):
        return data.encode(to_enc)
    else:
        return data

def conv_decoding( data ):
    lookup = ( 'ascii' , 'shift_jis','utf_8', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213',
            'shift_jis_2004','shift_jisx0213',
            'iso2022jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_3',
            'iso2022_jp_ext','latin_1' )
    if isinstance( data , unicode ):
        return data

    for encoding in lookup:
        try:
            data = data.decode('utf_8')
            #data = data.decode(encoding)
            return data
        except:
            pass

def SplitName( line ):
    p=line.find("//")
    if p != -1:
       line = line[:p]

    # 改行コードを削除
    line = line.strip().replace(" " , "\t")

    while "\t\t" in line:
        line = line.replace("\t\t" , "\t")

    pos = line.find( "\t" )
    if pos==-1 : return  "" , ""
    name = line[:pos].strip()
    value = line[pos+1:].strip()

    if len(value)==0:
        return  "" , ""

    return name , value


def LoadValues( filename , encoding="utf-8" ):
    values = {}
    for line in codecs.open( filename , "r"):
        name, value = SplitName( line )
        if name!="" and value!="":
            values[name] = value
    return values

def LoadValue( filename , valuename , type=str, encoding="utf-8" ):
    valuename = conv_decoding( valuename )
    for line in codecs.open( filename , "r" ):
        name, value = SplitName( line )
        if name == valuename:
           return type(value)

def Write( f , value ):
    if not hasattr( value , "__iter__" ):
        if isinstance(value,basestring):
            f.write( conv_decoding(value) + "\n" )
        else:
            f.write( str(value) + "\n" )
    else:
        for v in value:
            if isinstance(v,basestring):
                f.write( conv_decoding(v) + "\t" )
            else:
                f.write( str(v) + "\t" )
        f.write( "\n" )


def LoadValueArray( filename , valuename , type = float , encoding="utf-8" ):
	l = []
	data = LoadValue( filename , valuename , str ).split()
	for i in range( len(data) ):
		l.append( type(data[i]) )
	return l

def SaveValue( filename , valuename , value , encoding="utf-8" ):
    list = []
    bFound = False

    valuename = conv_decoding( valuename )

    if os.access( filename , os.F_OK ):
        for line in codecs.open( filename , "r"):
            list.append( line )

    f = codecs.open( filename , "w")
    for line in list:
        n,v = SplitName( line )
        if n == valuename :
            f.write( valuename + "\t" )
            Write( f , value )
            bFound = True
        else : f.write( line )

    if not bFound :
        f.write( valuename + "\t" )
        Write( f , value )



def LoadMatrix( filename , type = float , encoding="utf-8" ):
    list = []
    for line in codecs.open( filename , "r" ):
        if line.find("//")==0:
            continue
        items = []
        for i in line.split():
            if type==str:
                items.append( i )
            else:
                items.append( type( i ) )
        list.append( items )
    return list

def LoadArray( filename , type = float , encoding="utf-8" ):
    list = []
    for line in codecs.open( filename , "r"):
        if line.find("//")==0:
            continue
        line = line.replace( "\r\n" , "" )
        line = line.replace( "\n" , "" )
        if type==str:
            list.append( line )
        else:
            list.append( type(line) )
    return list

def SaveMatrix( mat , filename , encoding="utf-8" ):
    f = codecs.open( filename , "w"  )
    for line in mat:
        for i in line:
            f.write( i )
            f.write( "\t" )
        f.write( "\n" )
    f.close()

def SaveArray( arr , filename , encoding="utf-8" ):
    f = codecs.open( filename , "w")
    for i in arr:
        f.write( i )
        f.write( "\n" )
    f.close()

def SaveObject( obj , filename ):
    f = open( filename , "w" )
    pickle.dump( obj , f )
    f.close()

def LoadObject( filename ):
    f = open( filename , "r" )
    obj = pickle.load( f )
    f.close()
    return obj

def SaveMultiDimMatrix( mat , filename , encoding="utf-8" ):
    def SaveList( f , data , idx=[] ):
        if hasattr( data, "__iter__" ):
            for i in range(len(data)):
                SaveList( f , data[i] , idx+[i])
        else:
            for i in idx:
                f.write( i )
                f.write( "\t" )
            f.write( data )
            f.write( "\t\n" )
            print idx,data

    f = codecs.open( filename , "w"  )
    SaveList( f , mat )
    f.close()



def main():
    a = [ [ [1,3] , [4,5] ],[ [1,3] , [4,5] ]]
    SaveMultiDimMatrix( a , "aaa.txt" )



if __name__ == '__main__':
    main()
