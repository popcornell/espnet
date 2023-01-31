#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=6
skip_stages="-1"

tr_dsets=train
cv_dsets=dev
tt_dsets=

manifests_root=./data/lhotse
gss_dump_root=./exp/gss
ngpu=4  # set equal to the number of GPUs you have, used for GSS and ASR training

# gss config
max_batch_dur=120 # set accordingly to your GPU VRAM, here I used 40GB
nj_gss=6
cmd_gss=run.pl

log "$0 $*"
. utils/parse_options.sh

. ./db.sh || exit 1;
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# chime6 data directories, which are generated from ${chime5_corpus},
# to synchronize audio files across arrays and modify the annotation (JSON) file accordingly
chime6_corpus=${PWD}/CHiME6
chime7_corpus=${PWD}/CHiME7


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CHIME5}" ]; then
    log "Fill the value of 'CHIME5' of db.sh"
    exit 1
fi
if [ -z "${MIXER6}" ]; then
    log "Fill the value of 'MIXER6' of db.sh"
    exit 1
fi
if [ -z "${DIPCO}" ]; then
    log "Fill the value of 'DIPCO' of db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"

    if ! which wget >/dev/null; then
        echo "$0: wget is not installed."
        exit 1
    fi

    if [ -d "${DIPCO}/audio" ]; then
        echo "${DIPCO} already exists,
        exiting as I am assuming it has already been downloaded."
    else
        dipco_url="https://s3.amazonaws.com/dipco/DiPCo.tgz"

        log "Download and Untar DIPCO corpus from ${dipco_url}"
        mkdir -p ${DIPCO}
        if ! wget ${dipco_url} -O ${DIPCO}/dipco.tgz; then
            echo "$0: error executing wget ${dipco_url}"
            exit 1
        fi

        if ! tar -xf ${DIPCO}/dipco.tgz -C ${DIPCO} --strip-components=1 --warning=no-unknown-keyword; then
            echo "$0: error un-tarring archive ${DIPCO}/dipco.tgz"
            exit 1
        fi
    fi

    log "Download and Untar Mixer6 corpus"
    if [ -d "${MIXER6}/audio" ]; then
        echo "${MIXER6} already exists,
        exiting as I am assuming it has already been downloaded."
    else
        log "Please download Mixer6 from LDC."
        exit 1;
    fi

    if [ -d "${chime6_corpus}" ]; then
        echo "${chime6_corpus} already exists,
        exiting as I am assuming it has already been created."
    else
        log "Generate CHiME6 corpus from CHiME5"
        local/generate_chime6_data.sh \
            --cmd "${train_cmd}" \
            ${CHIME5} \
            ${chime6_corpus}
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! contains ${skip_stages} 2; then
    log "Stage 2: generate CHiME7 dataset"
    if [ -d "${chime7_corpus}" ]; then
        echo "${chime7_corpus} already exists"
        exit
    fi

    python local/generate_chime7_data.py \
        -c ${chime6_corpus} \
        -d ${DIPCO} \
        -m ${MIXER6} \
        -o ${chime7_corpus}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! contains ${skip_stages} 3; then
    log "Stage 3: Parse data to Lhotse manifests"
    for dset in chime6 dipco mixer6; do
        for part in train dev; do
        if [ ${dset} == dipco ] && [ ${part} == train ]; then
            continue # dipco has no train set
        fi
        echo "Creating lhotse manifests for ${dset} in ${manifests_root}/${dset}"
        python local/get_lhotse_manifests.py -c ${chime7_corpus} \
            -d ${dset} \
            -p ${part} \
            -o ${manifests_root} \
            --ignore_shorter 0.2
        done
    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! contains ${skip_stages} 4; then
    log "Stage 4: Running GSS"
    # check if GSS is installed, if not stop, user must manually install it
    if [ ! command -v gss &> /dev/null ]; then
        echo "GPU-based Guided Source Separation (GSS) could not be found,
        please refer to the README for how to install it. \n
        See also https://github.com/desh2608/gss for more informations."
        exit
    fi

    for dset in chime6 dipco mixer6; do
        for part in train dev; do
            if [ ${dset} == dipco ] && [ ${part} == train ]; then
                continue # dipco has no train set
            fi

            echo "Running Guided Source Separation for ${dset}/${part}, results will be in ${gss_dump_root}/${dset}/${part}"

            local/run_gss.sh \
                --manifests-dir ${manifests_root} \
                --dset-name ${dset} \
                --dset-part ${part} \
                --exp_dir ${gss_dump_root} \
                --cmd ${cuda_cmd} \
                --nj ${ngpu} \
                --max_batch_dur ${max_batch_dur}

            echo "Guided Source Separation processing for ${dset}/${part} was successful !"
            echo "Parsing the GSS output to lhotse manifests which will be placed in ${manifests_root}/${dset}/${part}"
        done
    done
fi


if [ ${stage} -le 5 ] && [ $stop_stage -ge 5 ]; then
    # Preparing ASR training and validation data;
    for dset in chime6 dipco; do
        for part in train dev; do
            if [ ${dset} == dipco ] && [ ${part} == train ]; then
                continue # dipco has no train set
            fi
        done
    done

    python local/data/gss2lhotse.py \
        -i ${gss_dump_root} \
        -o ${manifests_root}/gss/

    # parse gss output to kaldi manifests
    # train set
    tr_kaldi_manifests=()
    part=train
    mic=ihm
    for dset in chime6 mixer6; do
        for mic in ihm mdm; do
            if [ ${dset} == mixer6 ] && [ ${mic} == ihm ]; then
                continue # not used right now
            fi

            lhotse kaldi export -p ${manifests_root}/${dset}/${part}/${dset}-${mic}_recordings_${part}.jsonl.gz  ${manifests_root}/${dset}/${part}/${dset}-${mic}_supervisions_${part}.jsonl.gz data/kaldi/${dset}/${part}/${mic}

            ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${part}/${mic}/utt2spk \
                > data/kaldi/${dset}/${part}/${mic}/spk2utt
            ./utils/fix_data_dir.sh data/kaldi/${dset}/${part}/${mic}

            tr_kaldi_manifests+=( "data/kaldi/${dset}/$part/$mic" )
        done
    done

    echo ${tr_kaldi_manifests[@]}

    ./utils/combine_data.sh data/kaldi/train_all ${tr_kaldi_manifests[@]}

    # dev set ihm
    cv_kaldi_manifests_ihm=()
    part=dev
    mic=ihm
    for dset in chime6 dipco; do
        lhotse kaldi export -p ${manifests_root}/${dset}/${part}/${dset}-${mic}_recordings_${part}.jsonl.gz  ${manifests_root}/${dset}/${part}/${dset}-${mic}_supervisions_${part}.jsonl.gz data/kaldi/${dset}/${part}/${mic}

        ./utils/utt2spk_to_spk2utt.pl data/kaldi/${dset}/${part}/${mic}/utt2spk \
            > data/kaldi/${dset}/${part}/${mic}/spk2utt
        ./utils/fix_data_dir.sh data/kaldi/${dset}/${part}/${mic}

        cv_kaldi_manifests_ihm+=( "data/kaldi/${dset}/${part}/${mic}" )
    done

    echo ${cv_kaldi_manifests_ihm[@]}

    ./utils/combine_data.sh data/kaldi/dev_ihm_all ${cv_kaldi_manifests_ihm[@]}

    # dev set gss
    #dset_part=dev
    #mic=gss
    #for dset in chime6 dipco mixer6; do
    # lhotse kaldi export -p $manifests_root/$dset/$dset_part/$dset- data/kaldi/$dset/$dset_part/$mic
    # $cv_kaldi_manifests_gss+=" data/kaldi/$dset/$dset_part/$mic"
    #done
    #./utils/combine_data.sh data/kaldi/dev_gss $cv_kaldi_manifests_gss
fi


nlsyms=data/nlsyms.txt

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "stage 6: Create DUMMY non linguistic symbols file: ${nlsyms}"
    touch ${nlsyms}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

