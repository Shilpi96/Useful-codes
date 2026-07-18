PRO RH_GET_HDR, HDR, $
                FREQ, DATE, HMER, DEC, SREF, ROTP

; Lecture dans le header des parametres necessaires a init_malax_2d

    date   = intarr(3)
    chdate = sxpar(hdr,'DATE-OBS')
    reads, chdate, date, format='(i4,1x,i2,1x,i2)'
    dat=date(0)
    date(0) = date(2)
    date(2) = dat    
    freq   = sxpar(hdr,'FREQ')
    chhmer = sxpar(hdr,'HMER')
    chdec  = sxpar(hdr,'DECL')
    ihmer  = intarr(4)
    reads, chhmer, ihmer, format='(i2,1x,i2,1x,i2,1x,i2)'
    idec   = intarr(4)
    reads,  chdec, idec,  format='(i3,1x,i3,1x,i3,1x,i3)'

    hmer      = fltarr(3)
    hmer(0:2) = float  (ihmer(0:2))
    hmer(2)   = hmer(2) + float(ihmer(3)) / 100.

    dec      = fltarr  (3)
    dec(0:2) = float   (idec(0:2))
    dec(2)   = dec (2) + float(idec(3)) / 100.

    sref = 'SOLEIL'             ; 'soleil' ou 'autre'.
    rotp = 1
    return
end

PRO GT_BEAM_FAST_CSV, file, outname

a = mrdfits(file, 1, hdr)
t = a.time
n = n_elements(t)

RH_GET_HDR, hdr, freq, date, hmer, dec, sref, rotp
INIT_MALAX_2D, date, hmer, dec, sref, rotp

; preallocate (vectorized memory allocation)
major = dblarr(n)
minor = dblarr(n)
angle = dblarr(n)

; still required loop (cannot remove)
for i=0, n-1 do begin
    hmsc = mil_time(t[i])
    rh_ellipse_maille, para, freq, hmsc
    major[i] = para[0]
    minor[i] = para[1]
    angle[i] = para[2]
endfor

; write CSV in one vectorized block
openw, lun, outname, /get_lun

printf, lun, 'time,major,minor,angle'

for i=0, n-1 do begin
    printf, lun, t[i], major[i], minor[i], angle[i], $
        format='(F20.0,",",F12.6,",",F12.6,",",F10.4)'
endfor

free_lun, lun

END
