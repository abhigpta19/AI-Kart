<template>
  <div class="flex flex-col gap-4">
    <div class="flex flex-row items-center">
      <h1 class="text-4xl lg:text-5xl">Log Analysis System</h1>
    </div>
    <div class="text-gray-600 dark:text-gray-200">
      <p>
        Elevate security monitoring with our Log Data Analysis System. This innovative solution swiftly processes log data, providing instant compliance and security threat insights. Powered by a finely trained GPT-2 medium model, our system excels at prompt analysis. Seamlessly deployed through our user-friendly website, it's a cutting-edge tool to bolster your security strategies and stay ahead in threat detection.
      </p>
      <br>
      <p>

Our system's capabilities are amplified by our robust dataset. Leveraging advanced data augmentation techniques, we've expanded our dataset, generating 1000 data points in Q&A format. This augmentation process fine-tuned the hyperparameters of our GPT-2 medium model, enhancing its accuracy and predictive power. As a result, our system is primed to provide you with even more precise and insightful security assessments.
      </p>
    </div>
  </div>
</template>

<script>
import IconEmailSpam from "@/components/icons/IconEmailSpam.vue";

export default {
  name: "AppIntroduction",
  components: {IconEmailSpam},
}
</script>

<style scoped>
a {
  @apply text-purple-500;
}
</style>
