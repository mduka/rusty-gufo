// Struct fields and header constants of the FFmpeg API gufo-ffmpeg uses,
// compiled against the build's FFmpeg headers. No FFmpeg function is called
// here, so gufo never links FFmpeg; the functions are loaded at run time.
// FFmpeg keeps public struct layouts stable within a major version, and
// gufo-ffmpeg refuses libraries of another major version than these headers.

#include <errno.h>
#include <stdint.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>

const int gufo_averror_eof = AVERROR_EOF;
const int gufo_averror_exit = AVERROR_EXIT;
const int gufo_averror_eagain = AVERROR(EAGAIN);
const int gufo_averror_einval = AVERROR(EINVAL);
const int gufo_avseek_size = AVSEEK_SIZE;
const int gufo_avseek_force = AVSEEK_FORCE;
const int64_t gufo_av_nopts_value = AV_NOPTS_VALUE;
const int64_t gufo_av_time_base = AV_TIME_BASE;
const int gufo_avmedia_type_video = AVMEDIA_TYPE_VIDEO;
const int gufo_av_pix_fmt_rgb24 = AV_PIX_FMT_RGB24;
const int gufo_sws_bicubic = SWS_BICUBIC;
const int gufo_av_ef_explode = AV_EF_EXPLODE;
const unsigned gufo_libavformat_version_major = LIBAVFORMAT_VERSION_MAJOR;
const unsigned gufo_libavcodec_version_major = LIBAVCODEC_VERSION_MAJOR;
const unsigned gufo_libavutil_version_major = LIBAVUTIL_VERSION_MAJOR;
const unsigned gufo_libswscale_version_major = LIBSWSCALE_VERSION_MAJOR;

unsigned char** gufo_avio_buffer(AVIOContext* io) { return &io->buffer; }

void gufo_avformat_set_custom_io(AVFormatContext* format, AVIOContext* io) {
  format->pb = io;
  format->flags |= AVFMT_FLAG_CUSTOM_IO;
}
void gufo_avformat_set_interrupt(AVFormatContext* format,
                                 int (*callback)(void*), void* opaque) {
  format->interrupt_callback.callback = callback;
  format->interrupt_callback.opaque = opaque;
}
void gufo_avformat_set_limits(AVFormatContext* format, int max_streams,
                              int64_t probesize,
                              int64_t max_analyze_duration) {
  format->max_streams = max_streams;
  format->probesize = probesize;
  format->max_analyze_duration = max_analyze_duration;
}
unsigned gufo_avformat_nb_streams(const AVFormatContext* format) {
  return format->nb_streams;
}
int64_t gufo_avformat_duration(const AVFormatContext* format) {
  return format->duration;
}
AVStream* gufo_avformat_stream(const AVFormatContext* format, unsigned i) {
  return format->streams[i];
}

AVCodecParameters* gufo_avstream_codecpar(const AVStream* stream) {
  return stream->codecpar;
}
double gufo_avstream_avg_frame_rate(const AVStream* stream) {
  return av_q2d(stream->avg_frame_rate);
}
int64_t gufo_avstream_nb_frames(const AVStream* stream) {
  return stream->nb_frames;
}
int gufo_avcodecpar_codec_type(const AVCodecParameters* parameters) {
  return parameters->codec_type;
}
int gufo_avcodecpar_codec_id(const AVCodecParameters* parameters) {
  return parameters->codec_id;
}

void gufo_avcodec_set_limits(AVCodecContext* codec, int thread_count,
                             int64_t max_pixels, int err_recognition) {
  codec->thread_count = thread_count;
  codec->max_pixels = max_pixels;
  codec->err_recognition = err_recognition;
}

int gufo_avframe_width(const AVFrame* frame) { return frame->width; }
int gufo_avframe_height(const AVFrame* frame) { return frame->height; }
int gufo_avframe_format(const AVFrame* frame) { return frame->format; }
const uint8_t* const* gufo_avframe_data(const AVFrame* frame) {
  return (const uint8_t* const*)frame->data;
}
const int* gufo_avframe_linesize(const AVFrame* frame) {
  return frame->linesize;
}

int gufo_avpacket_stream_index(const AVPacket* packet) {
  return packet->stream_index;
}
