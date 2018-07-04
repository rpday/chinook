#pragma rtGlobals=3		// Use modern global access method and strict wave access.

Function ParseCube(strnew)
	string strnew
	Make/T/N=(ItemsInList(strnew," ")) dimstrs
	dimstrs[] = StringFromList(p,strnew," ")
	Variable dim_off = str2num(dimstrs[2])
	Variable dim_end = str2num(dimstrs[3])
	Variable axlen = str2num(dimstrs[4])
	Variable dim_delta = (dim_end-dim_off)/axlen
//	Print axlen
	Make/N=(3) dims
	dims = {dim_off,dim_delta,axlen}
	KillWaves dimstrs
	return dims

end


Function Load_TB_ARPES()
	getfilefolderinfo/D //query file folder
	
	newpath/O ARPES S_path
	
	String lead= "NewScan"
	Prompt lead, "Enter the common file lead string: "//Query the lead on all relevant files
	DoPrompt "Enter the Data Information", lead
	if (V_flag)
		return -1 //User aborted
	endif
	
	variable i = 0
	string fname
	
	
	string filelist = indexedfile(ARPES,-1,".TXT")
	fname = lead+"_params.txt"
	String paramfile = S_path+fname
	Make/N=(3) xdim
	Make/N=(3) ydim
	Make/N=(3) Edim
//	Variable I0,I1,I2
	//loadwave/Q/G/K =2/N=Parameters paramfile
	loadwave/Q/J/K =2/N=Parameters paramfile //Q suppresses normal messages in history area, J indicates file is delimited text format, K=2 treats columns as text
	wave/T Param=$"Parameters0"//Store the loaded wave as a textwave
	Make/N=(3) dimout
	Edim = ParseCube(Param[3])
	xdim= ParseCube(Param[4])
	ydim = ParseCube(Param[5])
	

	
	Make/N= (xdim[2],ydim[2],Edim[2]) $lead+"_wave"
	WAVE wav = $lead+"_wave"
	
	Variable step = 0
	//string filename
	for(step=0;step<Edim[2];step+=1)	
		//filename = S_path+lead+"_"+num2str(step)+".txt"
		LoadWave/Q/G/M   /N=tmp_ S_path+lead+"_"+num2str(step)+".txt"
		wave tmp_0
		wav[][][step] +=tmp_0(x)(y)
		Print WaveMax(wav)
	endfor						
	setscale/p x xdim[0],xdim[1],wav
	setscale/p y ydim[0],ydim[1],wav
	setscale/p z Edim[0],Edim[1],wav
	Close/A
end