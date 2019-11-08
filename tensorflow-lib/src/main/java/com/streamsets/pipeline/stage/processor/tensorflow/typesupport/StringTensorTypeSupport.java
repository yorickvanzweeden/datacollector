/*
 * Copyright 2018 StreamSets Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.streamsets.pipeline.stage.processor.tensorflow.typesupport;

import com.streamsets.pipeline.api.Field;
import com.streamsets.pipeline.api.impl.Utils;

import org.tensorflow.DataType;
import org.tensorflow.Tensor;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class StringTensorTypeSupport extends AbstractTensorDataTypeSupport<ByteBuffer, String> {

  @Override
  public ByteBuffer allocateBuffer(long[] shape) {
    return ByteBuffer.allocate(calculateCapacityForShape(shape));
  }

  @Override
  public Tensor<String> createTensor(long[] shape, ByteBuffer buffer) {
    byte[][] b = new byte[][] { buffer.array() };
    return Tensor.create(b, String.class);
  }

  @Override
  public void writeField(ByteBuffer buffer, Field field) {
    Utils.checkState(field.getType() == Field.Type.STRING, "Not a String scalar");
    buffer.put(field.getValueAsString().getBytes(StandardCharsets.UTF_8));
  }

  @Override
  public List<Field> createListField(Tensor<String> tensor, ByteBuffer stringBuffer) {
    tensor.writeTo(stringBuffer);
    byte[] bytes = stringBuffer.array();
    // Marker between strings would be unknown, thus output one combined string.
    return new ArrayList<Field>() {{
      add(Field.create(new String(bytes, StandardCharsets.UTF_8)));
    }};
  }

  @Override
  public Field createPrimitiveField(Tensor<String> tensor) {
    return Field.create(tensor.bytesValue());
  }

  @Override
  public DataType getDataType() {
    return DataType.STRING;
  }
}
