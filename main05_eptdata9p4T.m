clear;close all;clc;
%% Path for data recovered from twix
datapath   = 'MRDataRaw/Varian_9p4T/';
%% Scan settings
settings.sequence           = 3; %1 C-SE %2 for TSE %3 for MSME % 4 for bSSFP
settings.combineChannels    = 'adaptComb'; % 'adaptComb' %'SoS'
settings.denoised           = 'Net'; %'Net'; %'BM4D' 'Wavelet' 'Net' false
settings.unWrapping         = 'laplacian'; %pumo_ho
settings.savematfile        = false;
settings.writenifti         = false;
settings.display            = true;
settings.reconstruct_sigmaH = 'HEPT'; %HEPT %CREPT %SHEPT
%% Path for fid file
[filename, dirpath] = uigetfile({'*.*'});
fid_file    = strrep(dirpath, '\', '/');
fprintf('| Initializing reconstruction pipeline by passing header information...\n');
%% ===================================================================== %%
[opt_mr,opt_phase,EPTrecon] = VarianPipeline(fid_file,settings); return
%% Save Matlab format of Raw Data
if settings.savematfile
    fprintf('| Writing magnitude/phase MAT file...\n');
    save([datapath,'\EPTrecon'],'EPTrecon');
else
end
%% Display phase and magnitude of Raw Data
if settings.display
    sl_tra = ceil(size(EPTrecon.RawData.phase,3)/2);
    
    figure;montage(EPTrecon.RawData.phase(:,:,sl_tra,:),'DisplayRange',...
        [-pi pi]);colormap gray;axis off image;
    
    figure;montage(EPTrecon.RawData.mag(:,:,sl_tra,:),'DisplayRange',...
        [0 10e4]);colormap gray;axis off image;
    
    opt_phase_disp = reshape(EPTrecon.opt_phase,[EPTrecon.parameters.nPE,...
        EPTrecon.parameters.nRO,1,EPTrecon.parameters.nSlc]);
    figure;montage(opt_phase_disp,'DisplayRange',[-pi pi]);colormap gray;...
        axis off image;
    
    opt_mr_disp = reshape(EPTrecon.opt_mr,[EPTrecon.parameters.nPE,...
        EPTrecon.parameters.nRO,1,EPTrecon.parameters.nSlc]);
    figure;montage(opt_mr_disp,'DisplayRange',[0 10e4]);colormap gray;...
        axis off image;
else
end
clear opt_phase_disp opt_mr_disp nSlices;
%% Mask from phantom
mag = sum(EPTrecon.RawData.mag,4);
for nSlices = 3
    tem = squeeze(mag(:,:,nSlices));
    tem = abs(tem);
    tem = (1/max(tem(:)))*tem;
    tem = double(roipoly(tem));close all;
    Mask3 = tem;
end
ROI = repmat(Mask3,[1 1 EPTrecon.parameters.nSlc]);
%% Write Nifti files of phase and magnitude (!Optimized)
if settings.writenifti
    fprintf('| Writing magnitude/phase NIfTI file...\n');
    voxelSize  = [EPTrecon.parameters.FOV/EPTrecon.parameters.nPE  ...
        EPTrecon.parameters.FOV/EPTrecon.parameters.nRO EPTrecon.parameters.slicethickness];
    
    phase      = reshape(EPTrecon.RawData.phase,EPTrecon.parameters.nPE,EPTrecon.parameters.nRO,...
        EPTrecon.parameters.nSlc.*EPTrecon.parameters.nEcho);
    nii        = make_nii(phase,voxelSize);
    save_nii(nii,[datapath,'\rawPhase.nii.gz']);
    
    magnitude  = reshape(EPTrecon.RawData.mag,EPTrecon.parameters.nPE,EPTrecon.parameters.nRO,...
        EPTrecon.parameters.nSlc.*EPTrecon.parameters.nEcho);
    nii        = make_nii(magnitude,voxelSize);
    save_nii(nii,[datapath,'\rawMag.nii.gz']);
    
    nii        = make_nii(EPTrecon.opt_phase,voxelSize);
    save_nii(nii,[datapath,'\Phase.nii.gz']);
    
    nii        = make_nii(EPTrecon.opt_mr,voxelSize);
    save_nii(nii,[datapath,'\Mag.nii.gz']);
    
    nii        = make_nii(ROI,voxelSize);
    save_nii(nii,[datapath,'\bse_Mask.nii.gz']);
else
end
clear Mask3 magnitude mag phase;
%% Conductivity reconstruction methods
FilterParam. KernelSize = [1,1]; c = 0.08;
if strcmp(settings.reconstruct_sigmaH,'CREPT')
    % Conductivity Reconstruction using CREPT
    % Imaging parametrs and reconstruction constants/parameters
    PixelSize  = [EPTrecon.parameters.FOV/EPTrecon.parameters.nPE  ...
        EPTrecon.parameters.FOV/EPTrecon.parameters.nRO];
    mu0        = 4*pi*1e-7;                             % Free space permeability
    gamma      = 42.576e6;                              % Gyromagnetic ratio
    omega      = 2*pi*gamma*EPTrecon.parameters.fieldS; % Larmor frequency @9.4T
    f          = 2*mu0*omega;
    bnd_val    = 1;                                     % Boundary conductivity value
    sigmaH_raw = zeros(size(EPTrecon.opt_phase));
    sigmaH     = zeros(size(EPTrecon.opt_phase));
    for slice = 1:EPTrecon.parameters.nSlc
        TemMask          = mk_bnd_smooth(ROI(:,:,slice),2);
        FilterParam.Mask = TemMask;
        disp(['Reconstructing slice# ',num2str(slice)]);
        phaseTem        = EPTrecon.opt_phase(:,:,slice).*TemMask;
        phaseTem        = EPTrecon.opt_phase(:,:,slice)-mean(nonzeros(phaseTem(:)));
        phase_grad      = mk_grad2D(phaseTem,PixelSize);
        phase_laplacian = mk_lap2D(phaseTem,PixelSize);
        rho             = poisson_solver(f,PixelSize,-c,phase_grad,phase_laplacian,bnd_val,TemMask);
        %                 rho = axb_solv(TemMask,TemMask,PixelSize,EPTrecon.parameters.nPE,bnd_val,phase_grad,phase_laplacian,f,c);
        tem = 1./rho; tem = abs(tem);      tem(isnan(tem)|isinf(tem)) = 0;
        sigmaH_raw(:,:,slice) = tem; clear tem;
        tem                   = weighted_spatial_mean_filter(sigmaH_raw(:,:,slice),...
            TemMask,FilterParam);
        sigmaH(:,:,slice)     = tem;
    end
elseif strcmp(settings.reconstruct_sigmaH,'HEPT')
    sigmaH_raw   = zeros(size(EPTrecon.opt_phase));
    sigmaH       = zeros(size(EPTrecon.opt_phase));
    permittivity = zeros(size(EPTrecon.opt_phase));
    frequency    = 128;
    B1p          = double(complex(EPTrecon.opt_mr,EPTrecon.opt_phase)).*ROI;
    [sigmaH_raw,permittivity] = HEPT(B1p,EPTrecon.parameters.vox,frequency,'method','2D','kernel','7pt');
    for slice = 1:EPTrecon.parameters.nSlc
        TemMask           = mk_bnd_smooth(ROI(:,:,slice),2);
        FilterParam.Mask  = TemMask;
        sigmaH(:,:,slice) = weighted_spatial_mean_filter(sigmaH_raw(:,:,slice),...
            TemMask,FilterParam);
        permittivity(:,:,slice) = weighted_spatial_mean_filter(permittivity(:,:,slice),...
            TemMask,FilterParam);
    end
elseif strcmp(settings.reconstruct_sigmaH,'SHEPT')
    sigmaH_raw   = zeros(size(EPTrecon.opt_phase));
    sigmaH       = zeros(size(EPTrecon.opt_phase));
    permittivity = zeros(size(EPTrecon.opt_phase));
    frequency    = 128;
    B1p          = double(complex(EPTrecon.opt_mr,EPTrecon.opt_phase));
    [sigmaH_raw,permittivity] = SHEPT(B1p,EPTrecon.parameters.vox,frequency,'method','2D','kernel','7pt');
    for slice = 1:EPTrecon.parameters.nSlc
        TemMask           = mk_bnd_smooth(ROI(:,:,slice),2);
        FilterParam.Mask  = TemMask;
        sigmaH(:,:,slice) = weighted_spatial_mean_filter(sigmaH_raw(:,:,slice),...
            TemMask,FilterParam);
        permittivity(:,:,slice) = weighted_spatial_mean_filter(permittivity(:,:,slice),...
            TemMask,FilterParam);
    end
elseif strcmp(settings.reconstruct_sigmaH,'PCM')
    sigmaH_raw   = zeros(size(EPTrecon.opt_phase));
    sigmaH       = zeros(size(EPTrecon.opt_phase));
    frequency    = 128;
    B1p          = double(complex(EPTrecon.opt_mr,RT_phantom.opt_phase));
    sigmaH_raw   = PCM(angle(B1p),voxelsize,frequency,'maxiter',500,'tol',1e-9,'display',10);
    for slice = 1:EPTrecon.parameters.nSlc
        TemMask           = mk_bnd_smooth(ROI(:,:,slice),2);
        FilterParam.Mask  = TemMask;
        sigmaH(:,:,slice) = weighted_spatial_mean_filter(sigmaH_raw(:,:,slice),...
            TemMask,FilterParam);
    end
else
end
%% Display Results of EPT RT 3 Tesla Phantom
%%Erode mask
for slice = 1:size(sigmaH,3)
    ROI(:,:,slice) = mk_bnd_smooth(ROI(:,:,slice),5);
end
%%display reconstucted images
sigmaH_disp     = reshape(sigmaH.*ROI,EPTrecon.parameters.nPE,EPTrecon.parameters.nRO,1,EPTrecon.parameters.nSlc);
figure('color','white');
montage(sigmaH_disp,'DisplayRange',[0 2]);

sigmaH_raw_disp = reshape(sigmaH_raw.*ROI,EPTrecon.parameters.nPE,EPTrecon.parameters.nRO,1,EPTrecon.parameters.nSlc);
figure('color','white');
montage(sigmaH_raw_disp,'DisplayRange',[0 2]);
%%
sigma = mean(sigmaH,3);
figure;imagesc(sigma,[0 2]);colormap jet;
cmap = colormap; cmap = [0 0 0;cmap]; colormap(cmap);
axis off image;