function datacube = import_txt_slices(filenames)
%import_txt_slices imports .txt slice output from TB-ARPES and combines into a 3D
%datacube set of dimension (Kx_pix x Ky_pix x E_pix) for further processing
%in MATLAB.

[Filenames,Path]=uigetfile('*.txt', 'MultiSelect','on');
cd(Path);
n=length(Filenames);
if n == 1
    datacube = importdata(Filenames);
else
    sz = size(importdata(Filenames{1}));
    l = sz(1); m = sz(2);
    datacube = zeros(l,m,n);
    for k = 1:n
        datacube(:,:,k) = importdata(Filenames{k});
    end
end
