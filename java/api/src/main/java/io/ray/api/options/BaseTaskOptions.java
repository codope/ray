package io.ray.api.options;

import java.io.Serializable;
import java.util.Map;

/** The options class for RayCall or ActorCreation. */
public abstract class BaseTaskOptions implements Serializable {

  /**
   * Mutable access to the resources map after construction could lead to subtle bugs. Therefore we
   * keep the field private and expose an immutable view through {@link #getResources()}.
   */
  private final Map<String, Double> resources;

  public BaseTaskOptions() {
    this.resources = java.util.Collections.emptyMap();
  }

  public BaseTaskOptions(Map<String, Double> resources) {
    java.util.Map<String, Double> filtered = new java.util.HashMap<>();

    for (java.util.Map.Entry<String, Double> entry : resources.entrySet()) {
      if (entry.getValue() == null || entry.getValue().compareTo(0.0) < 0) {
        throw new IllegalArgumentException(
            String.format(
                "Resource values should be non negative. Specified resource: %s = %s.",
                entry.getKey(), entry.getValue()));
      }
      // A resource value should be an integer if it is greater than 1.0 (e.g. 3.0 is valid, 3.5 is
      // not).
      if (entry.getValue().compareTo(1.0) >= 0
          && entry.getValue().compareTo(Math.floor(entry.getValue())) != 0) {
        throw new IllegalArgumentException(
            String.format(
                "A resource value should be an integer if it is greater than 1.0. Specified resource: %s = %s.",
                entry.getKey(), entry.getValue()));
      }

      // Filter out zero-capacity resources.
      if (entry.getValue() != 0) {
        filtered.put(entry.getKey(), entry.getValue());
      }
    }

    // Wrap with unmodifiable map to guarantee immutability from outside.
    this.resources = java.util.Collections.unmodifiableMap(filtered);
  }

  /** Immutable view of required resources. */
  public Map<String, Double> getResources() {
    return resources;
  }
}
