from diffusers import Flux2KleinPipeline


def main():
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B")
    pipe.load_lora_weights("/mnt/disks/ai-vision-jorge-shared-disk-h100/git_storage/flux-klein-api/workspace/adaptors/hegre_extended_grok_v2_copy_000002750.safetensors")
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.save_pretrained("workspace/merged-flux2-klein-4B")

if __name__ == "__main__":
    main()